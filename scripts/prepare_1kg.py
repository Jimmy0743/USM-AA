"""Prepare 1000 Genomes data without tgnn dependency."""
import os
import sys
import json
import struct
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import pysam

# Chromosome list
CHROMOSOMES = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]


# ============ VCF Parsing (replacing tgnn.sci.parser.vcf_parsing) ============
def parse_vcf(vcf_path):
    """Open and return a pysam VariantFile."""
    return pysam.VariantFile(vcf_path)


def get_samples(vcf_path):
    """Get list of sample IDs from VCF file."""
    with pysam.VariantFile(vcf_path) as vcf:
        return list(vcf.header.samples)


def get_genotype(record, sample_name):
    """Get genotype tuple for a sample from a VCF record."""
    try:
        gt = record.samples[sample_name]['GT']
        if gt is None:
            return (0, 0)
        return gt
    except (KeyError, TypeError):
        return (0, 0)


# ============ JSONL Dataset Classes ============
class MMapIndexedJsonlDatasetBuilder:
    """Builder for memory-mapped JSONL dataset."""
    _HDR_MAGIC = b'JSONLIDX\x00\x00'

    def __init__(self, path: str):
        self._path = path
        self._bin_path = path
        self._idx_path = f"{path}.idx"
        self._file = open(self._bin_path, 'wb')
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
            # Header
            f.write(self._HDR_MAGIC)
            f.write(struct.pack('<Q', 1))  # version
            f.write(struct.pack('<B', 1))  # dtype code
            f.write(struct.pack('<Q', len(self._sizes)))  # num items
            f.write(struct.pack('<Q', len(self._sizes)))  # doc count

            # Sizes
            sizes32 = np.array(self._sizes, dtype=np.int32)
            f.write(sizes32.tobytes(order='C'))

            # Pointers
            if self._sizes:
                pointers = np.cumsum([0] + self._sizes[:-1], dtype=np.int64)
            else:
                pointers = np.array([], dtype=np.int64)
            f.write(pointers.tobytes(order='C'))

            # Doc idx
            doc_idx = np.arange(len(self._sizes), dtype=np.int64)
            f.write(doc_idx.tobytes(order='C'))


# ============ Processing Functions ============
def process_samples(task_args, vcf_dir):
    """Process a chunk of samples for a chromosome."""
    sample_ids, chrom = task_args
    vcf_file = f"{vcf_dir}/1kGP_high_coverage_Illumina.{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    
    samples = {}
    with pysam.VariantFile(vcf_file) as vcf:
        # Subset to specific samples
        vcf.subset_samples(sample_ids)
        
        for var in vcf.fetch(chrom):
            for name in sample_ids:
                if name not in samples:
                    samples[name] = {"chrom": chrom, "sample_id": name, "hap0": [], "hap1": []}

                sample = samples[name]
                gt = get_genotype(var, name)
                
                if gt in ((0, 0), (0,), (None, None)):
                    continue

                pos = var.pos - 1  # Convert to 0-based
                alleles = var.alleles
                if len(alleles) < 2:
                    continue
                ref, alt = alleles[0], alleles[1]
                
                # Skip if non-standard bases
                if not set(ref).issubset({"A", "C", "G", "T"}):
                    continue
                if not set(alt).issubset({"A", "C", "G", "T"}):
                    continue

                for hap in (0, 1):
                    if hap >= len(gt):
                        continue
                    if not gt[hap]:
                        continue

                    # SNP
                    if len(ref) == len(alt):
                        sample[f"hap{hap}"].append({
                            "seq": alt[0],
                            "cigar": pysam.CDIFF,
                            "position": pos
                        })
                        continue

                    # Complex variant: first base
                    cigar = pysam.CEQUAL if alt[0] == ref[0] else pysam.CDIFF
                    sample[f"hap{hap}"].append({
                        "seq": alt[0],
                        "cigar": cigar,
                        "position": pos
                    })

                    # DEL
                    if len(ref) > len(alt):
                        num_del = len(ref) - len(alt)
                        seq = ref[1:1 + num_del]
                        cigar = pysam.CDEL
                    # INS
                    else:
                        num_ins = len(alt) - len(ref)
                        seq = alt[1:1 + num_ins]
                        cigar = pysam.CINS

                    sample[f"hap{hap}"].append({
                        "seq": seq,
                        "cigar": cigar,
                        "position": pos + 1
                    })

    return samples


def process_wrapper(args):
    """Wrapper for multiprocessing."""
    idx, task_args, vcf_dir = args
    return idx, process_samples(task_args, vcf_dir)


def main():
    vcf_dir = "/path/to/1kgenomes/vcf"  # Directory containing 1KG VCF files
    save_dir = "/path/to/output/1kg"  # Output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get sample IDs from first chromosome
    vcf_file = f"{vcf_dir}/1kGP_high_coverage_Illumina.chr1.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    sample_ids = get_samples(vcf_file)
    print(f"Found {len(sample_ids)} samples")
    
    # Build task list
    chunk_size = 4
    tasks = []
    for chrom in CHROMOSOMES:
        for i in range(0, len(sample_ids), chunk_size):
            tasks.append((sample_ids[i:i + chunk_size], chrom))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Process with multiprocessing
    num_workers = max(1, int(cpu_count() * 0.8))
    builder = MMapIndexedJsonlDatasetBuilder(f"{save_dir}/1kg.jsonl")
    batch_size = 512
    
    # Prepare args for pool
    pool_args = [(idx, task, vcf_dir) for idx, task in enumerate(tasks)]
    
    with Pool(num_workers) as pool:
        for idx, output in tqdm(pool.imap_unordered(process_wrapper, pool_args), 
                                total=len(tasks), desc="Processing 1KG"):
            for sample in output.values():
                chrom = sample["chrom"]
                sample_id = sample["sample_id"]
                
                # Process hap0
                hap0 = sample["hap0"]
                for vi in range(0, len(hap0), batch_size):
                    builder.add_item({
                        "sample_id": sample_id,
                        "chrom": chrom,
                        "variants": hap0[vi:vi + batch_size],
                    })
                    builder.end_document()

                # Process hap1
                hap1 = sample["hap1"]
                for vi in range(0, len(hap1), batch_size):
                    builder.add_item({
                        "sample_id": sample_id,
                        "chrom": chrom,
                        "variants": hap1[vi:vi + batch_size],
                    })
                    builder.end_document()

    builder.finalize()


if __name__ == "__main__":
    main()
