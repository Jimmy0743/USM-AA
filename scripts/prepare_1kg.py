"""Prepare 1000 Genomes variant data for USM-AA pretraining.

This script processes 1000 Genomes Project VCF files to extract
variant information (SNPs, insertions, deletions) for each sample
and haplotype, outputting the data in JSONL format.


"""
import os
import pysam
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from data_utils import MMapIndexedJsonlDatasetBuilder


CHROMOSOMES = ["chr" + str(i) for i in range(1, 23)] + ["chrX"]

# Global variable for worker processes (set by initializer).
_VCF_DIR = None


def _init_worker(vcf_dir: str) -> None:
    """Initialize worker process with VCF directory path.
    
    Args:
        vcf_dir: Path to directory containing VCF files.
    """
    global _VCF_DIR
    _VCF_DIR = vcf_dir


def get_samples(vcf_path: str) -> list:
    """Get list of sample IDs from a VCF file.
    
    Args:
        vcf_path: Path to the VCF file.
        
    Returns:
        List of sample ID strings.
    """
    with pysam.VariantFile(vcf_path) as vcf:
        return list(vcf.header.samples)


def get_genotype(record: pysam.VariantRecord, sample_name: str) -> tuple:
    """Get genotype tuple for a sample from a VCF record.
    
    Args:
        record: pysam VariantRecord object.
        sample_name: Name of the sample.
        
    Returns:
        Genotype tuple (e.g., (0, 1) for heterozygous).
        Returns (0, 0) if genotype cannot be determined.
    """
    try:
        gt = record.samples[sample_name]['GT']
        if gt is None:
            return (0, 0)
        return gt
    except (KeyError, TypeError):
        return (0, 0)


def _process_task(args: tuple) -> tuple:
    """Process variants for a batch of samples on a chromosome.
    
    This function is designed to run in a worker process. It reads
    variants from a VCF file and extracts variant information for
    the specified samples.
    
    Args:
        args: Tuple of (task_index, (sample_ids, chromosome)).
        
    Returns:
        Tuple of (task_index, samples_dict) where samples_dict maps
        sample IDs to their variant data.
    """
    idx, (sample_ids, chrom) = args
    vcf_file = f"{_VCF_DIR}/1kGP_high_coverage_Illumina.{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    
    samples = {}
    with pysam.VariantFile(vcf_file) as vcf:
        vcf.subset_samples(sample_ids)
        
        for var in vcf.fetch(chrom):
            for name in sample_ids:
                if name not in samples:
                    samples[name] = {
                        "chrom": chrom,
                        "sample_id": name,
                        "hap0": [],
                        "hap1": []
                    }

                sample = samples[name]
                gt = get_genotype(var, name)
                
                # Skip reference genotypes
                if gt in ((0, 0), (0,), (None, None)):
                    continue

                pos = var.pos - 1  # Convert to 0-based coordinates
                alleles = var.alleles
                if len(alleles) < 2:
                    continue
                ref, alt = alleles[0], alleles[1]
                
                # Skip variants with non-standard bases
                if not set(ref).issubset({"A", "C", "G", "T"}):
                    continue
                if not set(alt).issubset({"A", "C", "G", "T"}):
                    continue

                for hap in (0, 1):
                    if hap >= len(gt):
                        continue
                    if not gt[hap]:
                        continue

                    # Handle SNP
                    if len(ref) == len(alt):
                        sample[f"hap{hap}"].append({
                            "seq": alt[0],
                            "cigar": pysam.CDIFF,
                            "position": pos
                        })
                        continue

                    # Handle complex variant: process first base
                    cigar = pysam.CEQUAL if alt[0] == ref[0] else pysam.CDIFF
                    sample[f"hap{hap}"].append({
                        "seq": alt[0],
                        "cigar": cigar,
                        "position": pos
                    })

                    # Handle deletion
                    if len(ref) > len(alt):
                        num_del = len(ref) - len(alt)
                        seq = ref[1:1 + num_del]
                        cigar = pysam.CDEL
                    # Handle insertion
                    else:
                        num_ins = len(alt) - len(ref)
                        seq = alt[1:1 + num_ins]
                        cigar = pysam.CINS

                    sample[f"hap{hap}"].append({
                        "seq": seq,
                        "cigar": cigar,
                        "position": pos + 1
                    })

    return idx, samples


def main() -> None:
    """Main entry point for 1000 Genomes data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare 1000 Genomes variant data for USM-AA pretraining"
    )
    parser.add_argument(
        "--vcf_dir",
        type=str,
        required=True,
        help="Directory containing 1KG VCF files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4,
        help="Number of samples to process per task (default: 4)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Number of variants per output document (default: 512)"
    )
    args = parser.parse_args()
    
    vcf_dir = args.vcf_dir
    save_dir = args.output_dir
    chunk_size = args.chunk_size
    batch_size = args.batch_size
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get sample IDs from first chromosome
    vcf_file = f"{vcf_dir}/1kGP_high_coverage_Illumina.chr1.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    sample_ids = get_samples(vcf_file)
    print(f"Found {len(sample_ids)} samples")
    
    # Build task list (sample chunks x chromosomes)
    tasks = []
    for chrom in CHROMOSOMES:
        for i in range(0, len(sample_ids), chunk_size):
            tasks.append((sample_ids[i:i + chunk_size], chrom))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Process tasks in parallel
    num_workers = int(cpu_count() * 0.8)
    print(f"Using {num_workers} workers")
    builder = MMapIndexedJsonlDatasetBuilder(f"{save_dir}/1kg.jsonl")
    
    with Pool(num_workers, initializer=_init_worker, initargs=(vcf_dir,)) as pool:
        results = pool.imap_unordered(_process_task, enumerate(tasks), chunksize=1)
        
        for idx, output in tqdm(results, total=len(tasks), desc="Processing 1KG"):
            for sample in output.values():
                chrom = sample["chrom"]
                sample_id = sample["sample_id"]
                
                # Write haplotype 0 variants in batches
                hap0 = sample["hap0"]
                for vi in range(0, len(hap0), batch_size):
                    builder.add_item({
                        "sample_id": sample_id,
                        "chrom": chrom,
                        "variants": hap0[vi:vi + batch_size],
                    })
                    builder.end_document()

                # Write haplotype 1 variants in batches
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
