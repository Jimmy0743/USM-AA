"""Common data utilities for data preparation scripts.

This module provides utilities for building and reading memory-mapped
FASTA and JSONL datasets with index files for fast random access.
"""
import os
import json
import struct
import numpy as np
from tqdm import tqdm


# IUPAC ambiguous nucleotide codes to standard bases mapping.
IUPAC_TRANSLATE = {
    "R": "A", "Y": "C", "S": "G", "W": "A",
    "K": "G", "M": "A", "B": "C", "D": "A",
    "H": "A", "V": "A"
}
IUPAC_TRANSLATOR = str.maketrans(IUPAC_TRANSLATE)


class FastaIndex:
    """Index builder and reader for FASTA files.
    
    This class provides methods to build and write index files for FASTA
    datasets, enabling fast random access to sequences.
    
    Attributes:
        _HDR_MAGIC: Magic bytes for index file header validation.
    """
    
    _HDR_MAGIC = b'FASTAIDX\x00\x00'

    @classmethod
    def build_index(cls, filename: str) -> None:
        """Build index file for a FASTA file.
        
        Scans the FASTA file to compute byte offsets and sizes for each
        sequence, then writes the index to a .idx file.
        
        Args:
            filename: Path to the FASTA file to index.
        """
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
        
        # Sync timestamps with source file
        bin_stat = os.stat(filename)
        os.utime(idx_filename, (bin_stat.st_atime, bin_stat.st_mtime))
        print(f"Index built: {idx_filename}, {len(sizes)} sequences")

    @classmethod
    def _write_index(cls, path: str, sizes: list) -> None:
        """Write index data to file.
        
        Args:
            path: Output path for the index file.
            sizes: List of byte sizes for each document.
        """
        with open(path, 'wb') as f:
            # Header
            f.write(cls._HDR_MAGIC)
            f.write(struct.pack('<Q', 1))  # version
            f.write(struct.pack('<B', 1))  # dtype code (uint8)
            f.write(struct.pack('<Q', len(sizes)))  # num items
            f.write(struct.pack('<Q', len(sizes)))  # doc count
            
            # Sizes array
            sizes32 = np.array(sizes, dtype=np.int32)
            f.write(sizes32.tobytes(order='C'))
            
            # Pointers array (cumulative offsets)
            pointers = np.cumsum([0] + sizes[:-1], dtype=np.int64) if sizes else np.array([], dtype=np.int64)
            f.write(pointers.tobytes(order='C'))
            
            # Document index array
            doc_idx = np.arange(len(sizes), dtype=np.int64)
            f.write(doc_idx.tobytes(order='C'))


class JsonlIndex:
    """Index builder for JSONL files.
    
    Attributes:
        _HDR_MAGIC: Magic bytes for index file header validation.
    """
    
    _HDR_MAGIC = b'JSONLIDX\x00\x00'

    @classmethod
    def _write_index(cls, path: str, sizes: list) -> None:
        """Write index data to file.
        
        Args:
            path: Output path for the index file.
            sizes: List of byte sizes for each document.
        """
        with open(path, 'wb') as f:
            # Header
            f.write(cls._HDR_MAGIC)
            f.write(struct.pack('<Q', 1))  # version
            f.write(struct.pack('<B', 1))  # dtype code
            f.write(struct.pack('<Q', len(sizes)))  # num items
            f.write(struct.pack('<Q', len(sizes)))  # doc count

            # Sizes array
            sizes32 = np.array(sizes, dtype=np.int32)
            f.write(sizes32.tobytes(order='C'))

            # Pointers array
            if sizes:
                pointers = np.cumsum([0] + sizes[:-1], dtype=np.int64)
            else:
                pointers = np.array([], dtype=np.int64)
            f.write(pointers.tobytes(order='C'))

            # Document index array
            doc_idx = np.arange(len(sizes), dtype=np.int64)
            f.write(doc_idx.tobytes(order='C'))


class MMapIndexedFastaDataset:
    """Memory-mapped FASTA dataset for efficient random access.
    
    Uses memory mapping to provide efficient random access to sequences
    in large FASTA files without loading the entire file into memory.
    
    Attributes:
        _path: Path to the FASTA file.
        verbose: Whether to print verbose output.
    
    Example:
        >>> dataset = MMapIndexedFastaDataset("/path/to/sequences.fasta")
        >>> print(len(dataset))
        1000
        >>> seq = dataset[0]
        >>> print(seq["id"], len(seq["seq"]))
    """

    def __init__(self, path: str, verbose: bool = False) -> None:
        """Initialize the dataset.
        
        Args:
            path: Path to the FASTA file.
            verbose: Whether to print verbose output.
            
        Raises:
            AssertionError: If the file does not exist.
        """
        self._path = path
        self.verbose = verbose
        assert os.path.isfile(path), f"File not found: {path}"
        self._do_init(path)

    def _do_init(self, path: str) -> None:
        """Initialize memory mapping and load index.
        
        Args:
            path: Path to the FASTA file.
        """
        idx_path = f"{path}.idx"
        if not os.path.isfile(idx_path):
            print(f"Building index: {idx_path}")
            FastaIndex.build_index(path)
        
        # Read index header
        with open(idx_path, 'rb') as f:
            magic = f.read(10)
            assert magic == FastaIndex._HDR_MAGIC, "Invalid index"
            f.read(8)  # version
            f.read(1)  # dtype
            self._len = struct.unpack('<Q', f.read(8))[0]
            f.read(8)  # doc_count
            offset = f.tell()
        
        # Memory map index arrays
        idx_mmap = np.memmap(idx_path, mode='r', order='C')
        self._sizes = np.frombuffer(idx_mmap, dtype=np.int32, count=self._len, offset=offset)
        self._pointers = np.frombuffer(idx_mmap, dtype=np.int64, count=self._len,
                                       offset=offset + self._sizes.nbytes)
        
        # Memory map data file
        self._bin_buffer = np.memmap(path, mode='r', dtype=np.uint8)

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self._len

    def __getitem__(self, idx: int) -> dict:
        """Get a sequence by index.
        
        Args:
            idx: Index of the sequence to retrieve.
            
        Returns:
            Dictionary with 'id' and 'seq' keys.
        """
        ptr = self._pointers[idx]
        size = self._sizes[idx]
        data = bytes(self._bin_buffer[ptr:ptr + size]).decode('utf-8')
        title, *lines = data.split("\n")
        title = title[1:].rstrip()
        seq = "".join(lines)
        seq_id = title.split(None, 1)[0]
        return {"id": seq_id, "seq": seq}
    
    def __iter__(self):
        """Iterate over all sequences in the dataset."""
        for i in range(len(self)):
            yield self[i]


class MMapIndexedFastaDatasetBuilder:
    """Builder for creating memory-mapped FASTA datasets.
    
    Provides a streaming interface for writing FASTA sequences and
    automatically building the index file upon finalization.
    
    Example:
        >>> builder = MMapIndexedFastaDatasetBuilder("/path/to/output.fasta")
        >>> builder.add_item({"id": "seq1", "seq": "ACGT"})
        >>> builder.end_document()
        >>> builder.finalize()
    """

    def __init__(self, path: str) -> None:
        """Initialize the builder.
        
        Args:
            path: Output path for the FASTA file.
        """
        self._path = path
        self._file = open(path, 'w', encoding='utf-8')
        self._sizes = []
        self._current_size = 0

    def add_item(self, data: dict) -> None:
        """Add a FASTA entry.
        
        Args:
            data: Dictionary with 'id' and 'seq' keys.
        """
        seq_id = data.get("id", "seq")
        seq = data.get("seq", "")
        content = f">{seq_id}\n{seq}\n"
        self._file.write(content)
        self._current_size += len(content.encode('utf-8'))

    def end_document(self) -> None:
        """Mark the end of the current document."""
        self._sizes.append(self._current_size)
        self._current_size = 0

    def finalize(self) -> None:
        """Finalize the dataset and build the index."""
        self._file.close()
        FastaIndex.build_index(self._path)
        print(f"Built FASTA dataset: {self._path}, {len(self._sizes)} sequences")


class MMapIndexedJsonlDataset:
    """Memory-mapped JSONL dataset for efficient random access.
    
    Attributes:
        _path: Path to the JSONL file.
        verbose: Whether to print verbose output.
    """

    def __init__(self, path: str, verbose: bool = False) -> None:
        """Initialize the dataset.
        
        Args:
            path: Path to the JSONL file.
            verbose: Whether to print verbose output.
            
        Raises:
            AssertionError: If the file or index does not exist.
        """
        self._path = path
        self.verbose = verbose
        assert os.path.isfile(path), f"File not found: {path}"
        self._do_init(path)

    def _do_init(self, path: str) -> None:
        """Initialize memory mapping and load index.
        
        Args:
            path: Path to the JSONL file.
        """
        idx_path = f"{path}.idx"
        assert os.path.isfile(idx_path), f"Index not found: {idx_path}"
        
        # Read index header
        with open(idx_path, 'rb') as f:
            magic = f.read(10)
            assert magic == JsonlIndex._HDR_MAGIC, "Invalid index"
            f.read(8)  # version
            f.read(1)  # dtype
            self._len = struct.unpack('<Q', f.read(8))[0]
            f.read(8)  # doc_count
            offset = f.tell()
        
        # Memory map index arrays
        idx_mmap = np.memmap(idx_path, mode='r', order='C')
        self._sizes = np.frombuffer(idx_mmap, dtype=np.int32, count=self._len, offset=offset)
        self._pointers = np.frombuffer(idx_mmap, dtype=np.int64, count=self._len,
                                       offset=offset + self._sizes.nbytes)
        
        # Memory map data file
        self._bin_buffer = np.memmap(path, mode='r', dtype=np.uint8)

    def __len__(self) -> int:
        """Return the number of documents in the dataset."""
        return self._len

    def __getitem__(self, idx: int) -> dict:
        """Get a document by index.
        
        Args:
            idx: Index of the document to retrieve.
            
        Returns:
            Parsed JSON document as a dictionary.
        """
        ptr = self._pointers[idx]
        size = self._sizes[idx]
        data = bytes(self._bin_buffer[ptr:ptr + size]).decode('utf-8')
        return json.loads(data.strip())
    
    def __iter__(self):
        """Iterate over all documents in the dataset."""
        for i in range(len(self)):
            yield self[i]


class MMapIndexedJsonlDatasetBuilder:
    """Builder for creating memory-mapped JSONL datasets.
    
    Example:
        >>> builder = MMapIndexedJsonlDatasetBuilder("/path/to/output.jsonl")
        >>> builder.add_item({"key": "value"})
        >>> builder.end_document()
        >>> builder.finalize()
    """

    def __init__(self, path: str) -> None:
        """Initialize the builder.
        
        Args:
            path: Output path for the JSONL file.
        """
        self._path = path
        self._idx_path = f"{path}.idx"
        self._file = open(self._path, 'wb')
        self._sizes = []
        self._current_size = 0
        self._buffer = []

    def add_item(self, data: dict) -> None:
        """Add a JSON item to the current document.
        
        Args:
            data: Dictionary to serialize as JSON.
        """
        line = json.dumps(data, ensure_ascii=False) + "\n"
        encoded = line.encode('utf-8')
        self._buffer.append(encoded)
        self._current_size += len(encoded)

    def end_document(self) -> None:
        """End current document and flush buffer to disk."""
        for chunk in self._buffer:
            self._file.write(chunk)
        self._sizes.append(self._current_size)
        self._current_size = 0
        self._buffer = []

    def finalize(self) -> None:
        """Finalize the dataset and build the index."""
        self._file.close()
        JsonlIndex._write_index(self._idx_path, self._sizes)
        print(f"Built JSONL dataset: {self._path}, {len(self._sizes)} documents")
