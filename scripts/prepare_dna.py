"""Prepare DNA data without tgnn dependency."""
import os
import json
import struct
import numpy as np
from tqdm import tqdm


class MMapIndexedJsonlDatasetBuilder:
    """Builder for memory-mapped JSONL dataset."""
    _HDR_MAGIC = b'JSONLIDX\x00\x00'

    def __init__(self, path: str):
        self._path = path
        self._idx_path = f"{path}.idx"
        self._file = open(self._path, 'wb')
        self._sizes = []
        self._current_size = 0
        self._buffer = []

    def add_item(self, data: dict):
        """Add a JSON item."""
        line = json.dumps(data, ensure_ascii=False) + "\n"
        encoded = line.encode('utf-8')
        self._buffer.append(encoded)
        self._current_size += len(encoded)

    def end_document(self):
        """End current document and write buffer."""
        for chunk in self._buffer:
            self._file.write(chunk)
        self._sizes.append(self._current_size)
        self._current_size = 0
        self._buffer = []

    def finalize(self):
        """Finalize and build index."""
        self._file.close()
        self._write_index()
        print(f"Built JSONL dataset: {self._path}, {len(self._sizes)} documents")

    def _write_index(self):
        """Write index file."""
        with open(self._idx_path, 'wb') as f:
            f.write(self._HDR_MAGIC)
            f.write(struct.pack('<Q', 1))  # version
            f.write(struct.pack('<B', 1))  # dtype code
            f.write(struct.pack('<Q', len(self._sizes)))  # num items
            f.write(struct.pack('<Q', len(self._sizes)))  # doc count

            sizes32 = np.array(self._sizes, dtype=np.int32)
            f.write(sizes32.tobytes(order='C'))

            if self._sizes:
                pointers = np.cumsum([0] + self._sizes[:-1], dtype=np.int64)
            else:
                pointers = np.array([], dtype=np.int64)
            f.write(pointers.tobytes(order='C'))

            doc_idx = np.arange(len(self._sizes), dtype=np.int64)
            f.write(doc_idx.tobytes(order='C'))


def main():
    data_dir = "/path/to/data"  # Data directory
    path = f"{data_dir}/dnabert/train.txt"  # Input: one DNA sequence per line
    builder = MMapIndexedJsonlDatasetBuilder(f"{data_dir}/train.jsonl")  # Output
    
    with open(path, "r") as f:
        for line in tqdm(f, desc="Processing DNA"):
            seq = line.strip()
            builder.add_item({"seq": seq})
            builder.end_document()
    
    builder.finalize()


if __name__ == '__main__':
    main()
