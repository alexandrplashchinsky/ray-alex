
import h5py
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



def make_json_safe(value):
    """Convert HDF5/numpy attr values into normal Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


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
    
    
    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
        data_context: Optional["DataContext"] = None,
    ) -> List[ReadTask]:
        
        read_tasks: List[ReadTask] = []
        
        
        
        
        
        
        return read_tasks