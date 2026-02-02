"""Prepare reference genome data."""
import os
import json
import struct
import numpy as np
from tqdm import tqdm


# ============ IUPAC Translation ============
IUPAC_TRANSLATE = {
    "R": "A", "Y": "C", "S": "G", "W": "A",
    "K": "G", "M": "A", "B": "C", "D": "A",
    "H": "A", "V": "A"
}
TRANSLATOR = str.maketrans(IUPAC_TRANSLATE)


# ============ Index Classes ============
class FastaIndex:
    """FASTA file index."""
    _HDR_MAGIC = b'FASTAIDX\x00\x00'

    @classmethod
    def build_index(cls, filename: str):
        """Build FASTA index."""
        sizes = []
        with open(filename, 'r', encoding="utf-8") as f:
            fasta_lines = []
            for line in tqdm(f, desc="Building FASTA index"):
                if line.startswith(">") and fasta_lines:
                    length = len("".join(fasta_lines).encode('utf-8'))
                    sizes.append(length)
                    fasta_lines = []
                fasta_lines.append(line)
            
            if fasta_lines:
                length = len("".join(fasta_lines).encode('utf-8'))
                sizes.append(length)

        idx_filename = f"{filename}.idx"
        cls._write_index(idx_filename, sizes)
        
        # Sync timestamps
        bin_stat = os.stat(filename)
        os.utime(idx_filename, (bin_stat.st_atime, bin_stat.st_mtime))
        print(f"Index built: {idx_filename}, {len(sizes)} sequences")

    @classmethod
    def _write_index(cls, path: str, sizes):
        """Write index file."""
        with open(path, 'wb') as f:
            # Header
            f.write(cls._HDR_MAGIC)
            f.write(struct.pack('<Q', 1))  # version
            f.write(struct.pack('<B', 1))  # dtype code (uint8)
            f.write(struct.pack('<Q', len(sizes)))  # num items
            f.write(struct.pack('<Q', len(sizes)))  # doc count
            
            # Sizes
            sizes32 = np.array(sizes, dtype=np.int32)
            f.write(sizes32.tobytes(order='C'))
            
            # Pointers
            pointers = np.cumsum([0] + sizes[:-1], dtype=np.int64) if sizes else np.array([], dtype=np.int64)
            f.write(pointers.tobytes(order='C'))
            
            # Doc idx
            doc_idx = np.arange(len(sizes), dtype=np.int64)
            f.write(doc_idx.tobytes(order='C'))


class MMapIndexedFastaDataset:
    """Memory-mapped FASTA dataset."""

    def __init__(self, path: str, verbose: bool = False):
        self._path = path
        self.verbose = verbose
        assert os.path.isfile(path), f"File not found: {path}"
        self._do_init(path)

    def _do_init(self, path: str):
        idx_path = f"{path}.idx"
        if not os.path.isfile(idx_path):
            print(f"Building index: {idx_path}")
            FastaIndex.build_index(path)
        
        # Read index
        with open(idx_path, 'rb') as f:
            magic = f.read(10)
            assert magic == FastaIndex._HDR_MAGIC, "Invalid index"
            f.read(8)  # version
            f.read(1)  # dtype
            self._len = struct.unpack('<Q', f.read(8))[0]
            f.read(8)  # doc_count
            offset = f.tell()
        
        idx_mmap = np.memmap(idx_path, mode='r', order='C')
        self._sizes = np.frombuffer(idx_mmap, dtype=np.int32, count=self._len, offset=offset)
        self._pointers = np.frombuffer(idx_mmap, dtype=np.int64, count=self._len,
                                       offset=offset + self._sizes.nbytes)
        
        # Map bin file
        self._bin_buffer = np.memmap(path, mode='r', dtype=np.uint8)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        ptr = self._pointers[idx]
        size = self._sizes[idx]
        data = bytes(self._bin_buffer[ptr:ptr + size]).decode('utf-8')
        title, *lines = data.split("\n")
        title = title[1:].rstrip()
        seq = "".join(lines)
        seq_id = title.split(None, 1)[0]
        return {"id": seq_id, "seq": seq}
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class MMapIndexedFastaDatasetBuilder:
    """Builder for memory-mapped FASTA dataset."""

    def __init__(self, path: str):
        self._path = path
        self._file = open(path, 'w', encoding='utf-8')
        self._sizes = []
        self._current_size = 0

    def add_item(self, data: dict):
        """Add a FASTA entry."""
        seq_id = data.get("id", "seq")
        seq = data.get("seq", "")
        content = f">{seq_id}\n{seq}\n"
        self._file.write(content)
        self._current_size += len(content.encode('utf-8'))

    def end_document(self):
        """End current document."""
        self._sizes.append(self._current_size)
        self._current_size = 0

    def finalize(self):
        """Finalize and build index."""
        self._file.close()
        FastaIndex.build_index(self._path)
        print(f"Built FASTA dataset: {self._path}, {len(self._sizes)} sequences")


def main():
    path = "/path/to/grch38.fasta"  # Input: GRCh38 reference genome FASTA
    data_dir = "/path/to/output"  # Output directory
    os.makedirs(data_dir, exist_ok=True)
    
    ds = MMapIndexedFastaDataset(path, verbose=False)
    builder = MMapIndexedFastaDatasetBuilder(f"{data_dir}/ref.fasta")
    
    for data in tqdm(ds, desc="Processing reference"):
        # Translate IUPAC codes to standard bases
        data["seq"] = data["seq"].translate(TRANSLATOR)
        builder.add_item(data)
        builder.end_document()
    
    builder.finalize()


if __name__ == "__main__":
    main()
