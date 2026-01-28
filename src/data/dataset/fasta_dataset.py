import os
from tqdm import tqdm
import numpy as np
from src.utils import print_rank_0
from .indexed_dataset import MMIndex, MMapIndexedDataset


class FastaIndex(MMIndex):
    """FASTA file index."""
    _HDR_MAGIC = b'FASTAIDX\x00\x00'

    @classmethod
    def build_index(cls, filename: str):
        """Build FASTA index."""
        sizes = []
        with open(filename, 'r', encoding="utf-8") as f:
            fasta_lines = []
            for line in tqdm(f, desc="Building index"):
                if line.startswith(">") and fasta_lines:
                    length = len("".join(fasta_lines).encode('utf-8'))
                    sizes.append(length)
                    fasta_lines = []
                fasta_lines.append(line)

            if fasta_lines:
                length = len("".join(fasta_lines).encode('utf-8'))
                sizes.append(length)

        idx_filename = f"{filename}.idx"
        with cls.writer(idx_filename, np.uint8) as index:
            index.write(sizes)

        bin_stat = os.stat(filename)
        os.utime(idx_filename, (bin_stat.st_atime, bin_stat.st_mtime))


class MMapIndexedFastaDataset(MMapIndexedDataset):
    """Memory-mapped FASTA dataset."""

    def read_index(self, path: str, skip_warmup: bool = True):
        if not os.path.isfile(path) or not FastaIndex.is_index(path):
            print_rank_0(f"Building index: {path}")
            FastaIndex.build_index(self.bin_file)
        return FastaIndex(path, skip_warmup=skip_warmup, verbose=self.verbose)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        fasta_line = data.tobytes().decode("utf-8")
        title, *lines = fasta_line.split("\n")
        title = title[1:].rstrip()
        seq = "".join(lines)
        seq_id = title.split(None, 1)[0]
        return {"id": seq_id, "seq": seq}
