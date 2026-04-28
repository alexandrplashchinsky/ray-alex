
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from itertools import product
from math import prod
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlsplit
import sys

import numpy as np
import pandas as pd
from ray.data._internal.util import _check_import
from ray.data.block import BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
import h5py



def make_json_safe(value):
    """Convert HDF5/numpy attr values into normal Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _create_read_fn(
    batch: list[dict[str, object]],
    chunk_shape: tuple
) -> Callable[[], Iterable[pd.DataFrame]]:
    
    def read_fn() -> Iterable[pd.DataFrame]:
        arrays = []
        array_shapes = []
        chunk_shapes = []
        dtypes = []
        full_chunk_slices = []
        full_paddings = []
        
        for row in batch:
            chunk_slices = []
            padding = []
            curr_chunk_shape = list(chunk_shape)
            for dim, (i, size, chunk) in enumerate(zip(row['chunk_index'], row['meta']['shape'], chunk_shape)):
                start = i * chunk
                stop = min((i + 1) * chunk, size)
                chunk_slices.append((start, stop))
                
                if start + chunk > size:
                    padding_slice = start + chunk - size
                    curr_chunk_shape[dim] = stop - start
                else:
                    padding_slice = 0
                padding.append(padding_slice)
            full_chunk_slices.append(chunk_slices)
            arrays.append(row['array'])
            array_shapes.append(row['meta']['shape'])
            chunk_shapes.append(tuple(curr_chunk_shape))
            dtypes.append(row['meta']['dtype'])
            full_paddings.append(padding)
        
        yield pd.DataFrame({
            "array": arrays,
            "array_shape": array_shapes,
            "chunk_shape": chunk_shapes,
            "dtype": dtypes,
            "chunk_slices": full_chunk_slices,
            "padding": full_paddings
        })
    
    return read_fn


class HDF5Datasource(Datasource):
    
    def __init__(
        self, 
        path: str,
        chunk_shape: List[int],
        array_paths: List[str] | None = None,
    ):
        
        super().__init__()
        
        for val in chunk_shape:
            if val <= 0 or not isinstance(val, int):
                raise ValueError("chunk shape must only contain positive integerse")
        
        self.paths = [str(path)]
        self.chunk_shape = tuple(chunk_shape)
        self._metadata = self._load_consolidated_metadata()
        self._selected_arrays = self._select_array_metadata(array_paths)
        self._grid_shape_dict = self._gen_grid_shape()
    
    def _gen_grid_shape(self):
        
        grid_shape_dict = {}
        for array, meta in self._selected_arrays.items():
            shape = tuple(meta['shape'])
            
            if len(shape) != len(self.chunk_shape):
                raise ValueError(f"chunk shape must have same dimension length as the array: {array}")

            grid_shape = tuple(
                math.ceil(size / chunk)
                for size, chunk in zip(shape, self.chunk_shape)
            )
            
            grid_shape_dict[array] = {"meta": meta, "grid_shape": grid_shape}
            
        return grid_shape_dict

    def _select_array_metadata(
        self,
        array_paths: Iterable[str] | None
    ):
        arrays: dict[str, dict[str, object]] = {}
        for key, value in self._metadata.items():
            if value['type'] == 'dataset':
                arrays[key] = value
        
        if array_paths is None:
            selected_paths = [p for p in arrays.keys()]
        else:
            selected_paths = [p for p in arrays.keys() if p in array_paths]
            missing = [p for p in selected_paths if p not in arrays]
            if missing:
                available = ", ".join(sorted(p or "." for p in arrays.keys()))
                raise ValueError(
                    f"Array(s) not found: {', '.join(missing)}. Available: {available}"
                )
        
        return {path: arrays[path] for path in selected_paths}

    def _load_consolidated_metadata(self):
        metadata = {}
        
        with h5py.File(self.paths[0], "r") as f:
            metadata['/'] = {
                "type": "file",
                "attrs": {
                    key: make_json_safe(value)
                    for key, value in f.attrs.items()
                },
                "children": list(f.keys())
            }
            
            def visitor(name, obj):
                nonlocal metadata
                path = "/" + name
                
                if isinstance(obj, h5py.Group):
                    metadata[path] = {
                        "type": "group",
                        "attrs": {
                            key: make_json_safe(value)
                            for key, value in obj.attrs.items()
                        },
                        "children": list(obj.keys())
                    }
                
                elif isinstance(obj, h5py.Dataset):
                    metadata[path] = {
                        "type": "dataset",
                        "attrs": {
                            key: make_json_safe(value)
                            for key, value in obj.attrs.items()
                        },
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                        "ndim": obj.ndim,
                        "size": obj.size,
                        "chunks": obj.chunks,
                        "compression": obj.compression,
                        "compression_opts": obj.compression_opts
                    }
                
            f.visititems(visitor)
        
        return metadata
    
    
    def estimate_inmemory_data_size(self) -> Optional[int]:
        full_bytes_estimate = 0
        for _, meta in self._selected_arrays.items():
            shape = tuple(meta['shape'])
            dtype = np.dtype(meta["dtype"])
            bytes_estimate = int(prod(shape) * dtype.itemsize)
            full_bytes_estimate += bytes_estimate
        
        return full_bytes_estimate
    
    def _sizeof_batch(self, obj, seen=None):
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            size += sum(self._sizeof_batch(k, seen) + self._sizeof_batch(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(self._sizeof_batch(x, seen) for x in obj)

        return size
    
    
    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
        data_context: Optional["DataContext"] = None,
    ) -> List[ReadTask]:
        
        read_tasks: List[ReadTask] = []
        batch: list[dict[str, object]] = []
        
        num_chunks = sum(prod(value['grid_shape']) for _, value in self._grid_shape_dict.items())
        parallelism = min(parallelism, num_chunks) if num_chunks > 0 else 1
        batch_size = math.ceil(num_chunks / parallelism)
        
        for array, data in self._grid_shape_dict.items():
            for chunk_index in product(*(range(n) for n in data['grid_shape'])):
                
                batch.append({"array": array, "meta": data['meta'], "chunk_index": chunk_index})
                
                if len(batch) >= batch_size:
                    read_tasks.append(
                        ReadTask(
                            _create_read_fn(
                                batch,
                                self.chunk_shape
                            ),
                            BlockMetadata(
                                num_rows = len(batch),
                                size_bytes = self._sizeof_batch(batch),
                                input_files = [self.paths[0]],
                                exec_stats = None
                            )
                        )
                    )
                    batch = []
        if batch:
            read_tasks.append(
                ReadTask(
                    _create_read_fn(
                        batch,
                        self.chunk_shape
                    ),
                    BlockMetadata(
                        num_rows=len(batch),
                        size_bytes=self._sizeof_batch(batch),
                        input_files=[self.paths[0]],
                        exec_stats=None
                    )
                )
            )
        
        return read_tasks