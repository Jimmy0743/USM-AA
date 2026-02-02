import struct
import os
from functools import lru_cache
import numpy as np
from torch.utils.data import Dataset


class MMIndex:
    """Memory-mapped index."""
    _HDR_MAGIC = b'MMIDIDX\x00\x00'

    dtypes = {
        1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
        5: np.int64, 6: np.float32, 7: np.double, 8: np.uint16
    }

    @classmethod
    def code(cls, dtype):
        for k, v in cls.dtypes.items():
            if v == dtype:
                return k
        raise ValueError(f"Unknown dtype: {dtype}")

    @classmethod
    def is_index(cls, filename: str) -> bool:
        if not os.path.exists(filename):
            return False
        with open(filename, "rb") as f:
            magic = f.read(len(cls._HDR_MAGIC))
        return magic == cls._HDR_MAGIC

    @classmethod
    def writer(cls, path: str, dtype):
        class _Writer:
            def __enter__(self):
                self._file = open(path, 'wb')
                return self

            def write(self, sizes, doc_idx=None):
                if doc_idx is None:
                    doc_idx = np.arange(len(sizes), dtype=np.int64)
                else:
                    doc_idx = np.array(doc_idx, dtype=np.int64)

                # Write header
                self._file.write(cls._HDR_MAGIC)
                self._file.write(struct.pack('<Q', 1))
                self._file.write(struct.pack('<B', cls.code(dtype)))
                self._file.write(struct.pack('<Q', len(sizes)))
                self._file.write(struct.pack('<Q', len(doc_idx)))

                # Write sizes
                sizes32 = np.array(sizes, dtype=np.int32)
                self._file.write(sizes32.tobytes(order='C'))

                # Write pointers
                pointers = np.array(sizes, dtype=np.int64)
                pointers *= dtype().itemsize
                np.cumsum(pointers, axis=0, out=pointers)
                pointers = np.concatenate([[0], pointers[:-1]]) if len(pointers) > 0 else pointers
                self._file.write(pointers.tobytes(order='C'))

                # Write doc_idx
                self._file.write(doc_idx.tobytes(order='C'))

            def __exit__(self, *args):
                self._file.close()

        return _Writer()

    def __init__(self, path: str, skip_warmup: bool = True, verbose: bool = False):
        self.verbose = verbose
        self._do_init(path, skip_warmup)

    def _do_init(self, path: str, skip_warmup: bool):
        with open(path, 'rb') as f:
            magic = f.read(len(self._HDR_MAGIC))
            assert magic == self._HDR_MAGIC, "Invalid index file"

            version = struct.unpack('<Q', f.read(8))
            assert version == (1,)

            dtype_code = struct.unpack('<B', f.read(1))[0]
            self._dtype = self.dtypes[dtype_code]
            self._dtype_size = self._dtype().itemsize
            self._len = struct.unpack('<Q', f.read(8))[0]
            self._doc_count = struct.unpack('<Q', f.read(8))[0]
            offset = f.tell()

        self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
        self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                       offset=offset + self._sizes.nbytes)
        self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                      offset=offset + self._sizes.nbytes + self._pointers.nbytes)
        self.buffer_size = np.sum(self._sizes, dtype=np.int64)

    def __del__(self):
        if hasattr(self, '_bin_buffer_mmap') and self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

    @property
    def dtype(self):
        return self._dtype

    @property
    def sizes(self):
        return self._sizes

    @property
    def pointers(self):
        return self._pointers

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self._pointers[i], self._sizes[i]

    def __len__(self):
        return self._len


class MMapIndexedDataset(Dataset):
    """Memory-mapped indexed dataset base class."""

    def __init__(self, path: str, skip_warmup: bool = True, verbose: bool = False):
        super().__init__()
        self._path = path
        self.verbose = verbose
        assert os.path.isfile(path), f"File not found: {path}"
        self._do_init(path, skip_warmup)

    @property
    def index_file(self):
        return f"{self._path}.idx"

    @property
    def bin_file(self):
        return self._path

    def _do_init(self, path: str, skip_warmup: bool = True):
        self._index = self.read_index(self.index_file, skip_warmup)
        self._bin_buffer_mmap = np.memmap(self.bin_file, mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def read_index(self, path: str, skip_warmup: bool = True):
        raise NotImplementedError

    def __del__(self):
        if hasattr(self, '_bin_buffer_mmap') and self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ptr, size = self._index[idx]
        return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
