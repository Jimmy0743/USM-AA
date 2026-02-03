"""
Prepare DNA data with alignment using minimap2/mappy.
This script aligns DNA sequences to a reference genome and outputs
the alignments with CIGAR strings in JSONL format.
"""
import os
import re
import pysam
import mappy as mp
from tqdm import tqdm
from data_utils import MMapIndexedJsonlDatasetBuilder


# Regex pattern for parsing minimap2 cs (CIGAR-like) strings.
CS_PATTERN = (
    r'(?P<M>:(\d+))|'
    r'(?P<X>\*[\w~][\w~])|'
    r'(?P<I>\+[acgtnryswkmbdhvACGTNRYSWKMBDHV]+)|'
    r'(?P<D>\-[acgtnryswkmbdhvACGTNRYSWKMBDHV]+)'
)


class MinimapAligner:
    """Minimap2 aligner wrapper using the mappy library.
    
    Provides a convenient interface for aligning DNA sequences to a
    reference genome and parsing the resulting alignments.
    
    Attributes:
        aligner: The underlying mappy.Aligner instance.
    
    Preset options:
        - "sr": Illumina short read
        - "map-pb": PacBio
        - "map-ont": Oxford Nanopore
        - "splice": RNA-seq splice-aware
        - "asm5": Assembly to assembly (error ~5%)
        - "asm10": Assembly to assembly (error ~10%)
    """

    def __init__(self, fasta_file: str, preset: str = "sr", best_n: int = 1) -> None:
        """Initialize the aligner."""
        assert os.path.exists(fasta_file), f"Reference file not found: {fasta_file}"
        self.aligner = mp.Aligner(fasta_file, preset=preset, best_n=best_n)

    def __call__(self, *args, **kwargs) -> list:
        """Alias for the map method."""
        return self.map(*args, **kwargs)

    def map(self, query_seq: str, min_mapq: int = 0, sort: str = "mapq") -> list:
        """Align a query sequence to the reference genome.
        
        Args:
            query_seq: DNA sequence to align.
            min_mapq: Minimum mapping quality threshold.
            sort: Sort method for hits ("mapq" or "mlen").
        
        Returns:
            List of alignment dictionaries containing:
                - seq: The aligned sequence
                - strand: 0 for forward, 1 for reverse complement
                - chrom: Chromosome name
                - ref_start: Reference start position
                - ref_end: Reference end position
                - query_start: Query start position
                - query_end: Query end position
                - mapq: Mapping quality
                - mlen: Aligned length
                - cigars: List of CIGAR operations
                - positions: Reference positions for each operation
        
        Raises:
            ValueError: If an unsupported sort method is specified.
        """
        alignments = []
        hits = self.aligner.map(query_seq, cs=True)
        
        # Sort hits by specified criterion
        sort = sort.lower()
        if sort == "mlen":
            hits = sorted(hits, key=lambda h: h.mlen, reverse=True)
        elif sort == "mapq":
            hits = sorted(hits, key=lambda h: h.mapq, reverse=True)
        else:
            raise ValueError(f"Sort method '{sort}' is not supported")

        for hit in hits:
            if hit is None or hit.mapq < min_mapq:
                continue

            chrom = hit.ctg
            query_start = hit.q_st
            query_end = hit.q_en
            
            # Handle reverse complement alignment
            if hit.strand == -1:
                query_seq = mp.revcomp(query_seq)
                query_start, query_end = len(query_seq) - query_end, len(query_seq) - query_start

            rpos = hit.r_st
            qpos = 0
            cigars = []
            positions = []
            
            # Add soft clip at start if needed
            if query_start > 0:
                length = query_start
                cigars.append([length, pysam.CSOFT_CLIP, query_seq[qpos:qpos + length]])
                positions.append(rpos - length)
                qpos += length

            # Parse CIGAR operations from cs string
            tokens = re.finditer(CS_PATTERN, hit.cs)
            for m in tokens:
                if m.group('M'):  # Match: :30 -> 30=
                    length = int(m.group(2))
                    if length:
                        cigars.append((length, pysam.CEQUAL, query_seq[qpos:qpos + length]))
                        positions.append(rpos)
                        qpos += length
                        rpos += length
                elif m.group('X'):  # Mismatch: *ag -> 1X
                    length = 1
                    cigars.append((length, pysam.CDIFF, m.group('X')[-1].upper()))
                    positions.append(rpos)
                    qpos += length
                    rpos += length
                elif m.group('I'):  # Insertion: +atg -> 3I
                    seq = m.group('I')[1:].upper()
                    length = len(seq)
                    cigars.append((length, pysam.CINS, seq))
                    positions.append(rpos)
                    qpos += length
                elif m.group('D'):  # Deletion: -ctg -> 3D
                    seq = m.group('D')[1:].upper()
                    length = len(seq)
                    cigars.append((length, pysam.CDEL, seq))
                    positions.append(rpos)
                    rpos += length

            # Add soft clip at end if needed
            if query_end < len(query_seq):
                assert qpos == query_end
                length = len(query_seq) - query_end
                cigars.append([length, pysam.CSOFT_CLIP, query_seq[qpos:qpos + length]])
                positions.append(rpos)

            # Verify CIGAR length matches query length
            cigar_len = sum([c[0] for c in cigars if c[1] not in (pysam.CDEL,)])
            assert cigar_len == len(query_seq), \
                f"CIGAR length mismatch: {cigars}, seq_len={len(query_seq)}"

            align = {
                "seq": query_seq,
                "strand": 1 if hit.strand == -1 else 0,
                "chrom": chrom,
                "ref_start": hit.r_st,
                "ref_end": hit.r_en,
                "query_start": query_start,
                "query_end": query_end,
                "mapq": hit.mapq,
                "mlen": hit.mlen,
                "cigars": cigars,
                "positions": positions
            }
            alignments.append(align)

        return alignments


def main() -> None:
    """Main entry point for DNA data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare DNA data with alignment using minimap2"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input DNA sequences file (one sequence per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=True,
        help="Reference genome FASTA file path"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="sr",
        help="Minimap2 preset (default: sr)"
    )
    parser.add_argument(
        "--best_n",
        type=int,
        default=10,
        help="Number of best alignments to keep (default: 10)"
    )
    args = parser.parse_args()
    
    # Initialize aligner
    print(f"Loading reference: {args.ref}")
    aligner = MinimapAligner(args.ref, preset=args.preset, best_n=args.best_n)
    
    # Initialize dataset builder
    builder = MMapIndexedJsonlDatasetBuilder(args.output)
    
    # Process sequences
    print(f"Processing: {args.input}")
    with open(args.input, "r") as f:
        for line in tqdm(f, desc="Aligning sequences"):
            sequence = line.strip()
            if not sequence:
                continue
            alignments = aligner(sequence)
            builder.add_item({"seq": sequence, "alignments": alignments})
            builder.end_document()
    
    builder.finalize()
    print("Done!")


if __name__ == '__main__':
    main()
