import os
import json
from tqdm import tqdm
import numpy as np
from src.utils import print_rank_0
from .indexed_dataset import MMIndex, MMapIndexedDataset


class JsonlIndex(MMIndex):
    """JSONL file index."""
    _HDR_MAGIC = b'JSONIDX\x00\x00'

    @classmethod
    def build_index(cls, filename: str):
        """Build JSONL index."""
        assert filename.endswith("jsonl"), f"file is not jsonl: {filename}"

        sizes = []
        with open(filename, 'r', encoding="utf-8") as f:
            for line in tqdm(f, desc="Building index"):
                sizes.append(len(line.encode('utf-8')))

        idx_filename = f"{filename}.idx"
        with cls.writer(idx_filename, np.uint8) as index:
            index.write(sizes)

        # Sync timestamps
        import shutil
        bin_stat = os.stat(filename)
        os.utime(idx_filename, (bin_stat.st_atime, bin_stat.st_mtime))


class MMapIndexedJsonlDataset(MMapIndexedDataset):
    """Memory-mapped JSONL dataset."""

    def read_index(self, path: str, skip_warmup: bool = True):
        if not os.path.isfile(path) or not JsonlIndex.is_index(path):
            print_rank_0(f"Building index: {path}")
            JsonlIndex.build_index(self.bin_file)
        return JsonlIndex(path, skip_warmup=skip_warmup, verbose=self.verbose)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return json.loads(data.tobytes())
