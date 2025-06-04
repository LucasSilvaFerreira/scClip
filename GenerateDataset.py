import pandas as pd
import scipy.io
import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


df_bar = pd.read_csv('GSM8015425_3T3_K562_ATAC_barcodes_human.tsv', sep='\t', header=None)
df_feature = pd.read_csv('GSM8015425_3T3_K562_ATAC_features_human.tsv', sep='\t', header=None)
df_feature
mtx_file = 'GSM8015425_3T3_K562_ATAC_peak_count_matrix_human.mtx'
sparse_matrix = scipy.io.mmread(mtx_file)
df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix)
df.columns = df_bar[0]
df.index = df_feature[0]
df = df.T
binary_df = df.gt(0).astype('Sparse[int]')




GENOME = 'hg38.fa'
genome = Fasta(GENOME)


n_samples = 100


# GENOME = 'hg38.fa'
# genome = Fasta(GENOME)

def extract_top_binary_patterns(binary_df, chrom: str, start: int, end: int,
                                window_size: int = 50000, bin_size: int = 500,
                                top_n: int = 10, plot: bool = False):
    """
    Extract most frequent binary patterns in a region around a coordinate.

    Returns:
        List[Tuple[str, int]]: Top N binary patterns and their frequencies, padded with zeros if needed.
    """
    def parse_coords(col_name):
        match = re.match(r"(chr[^\-]+)-(\d+)-(\d+)", col_name)
        if match:
            c, s, e = match.groups()
            center = (int(s) + int(e)) // 2
            return c, center
        return None, None

    parsed = [parse_coords(col) for col in binary_df.columns]
    coord_df = pd.DataFrame(parsed, columns=["chrom", "center"])
    coord_df["colname"] = binary_df.columns

    center = (start + end) // 2
    half_window = window_size // 2
    win_start, win_end = center - half_window, center + half_window
    bins = np.arange(win_start, win_end, bin_size)
    num_bins = len(bins)

    in_window = coord_df[
        (coord_df["chrom"] == chrom) &
        (coord_df["center"] >= win_start) &
        (coord_df["center"] < win_end)
    ].copy()

    if in_window.empty:
        return [('0' * num_bins, 0)] * top_n

    in_window["bin"] = ((in_window["center"] - win_start) // bin_size).astype(int)

    bin_matrix = np.zeros((binary_df.shape[0], num_bins), dtype=int)

    for _, row in in_window.iterrows():
        col_idx = binary_df.columns.get_loc(row["colname"])
        bin_matrix[:, row["bin"]] += binary_df.iloc[:, col_idx].values

    bin_matrix = (bin_matrix > 0).astype(int)

    patterns = ["".join(map(str, row)) for row in bin_matrix]
    pattern_counts = Counter(patterns)
    top_patterns = pattern_counts.most_common(top_n)

    # Pad with zero patterns if needed
    if len(top_patterns) < top_n:
        top_patterns += [('0' * num_bins, 0)] * (top_n - len(top_patterns))

    if plot:
        plt.figure(figsize=(10, 6))
        for i, (pattern, count) in enumerate(top_patterns):
            plt.barh(i, count)
            plt.text(count + 2, i, pattern, va='center', fontsize=8)
        plt.yticks(range(top_n), [f'Pattern {i+1}' for i in range(top_n)])
        plt.xlabel("Frequency")
        plt.title(f"Top {top_n} Binary Patterns around {chrom}:{center:,} (±{half_window:,}bp)")
        plt.tight_layout()
        plt.show()

    return top_patterns



hot_encoder = dict(zip('ACTGN'[::], np.eye(5)))
def convert_seq(seq):

  return np.array([ hot_encoder[s] for s in seq.upper()]).T


def get_sequence_from_fasta( chrom: str, start: int, end: int) -> str:
    """
    Extracts a DNA sequence from a FASTA file given a genomic coordinate.

    Args:
        chrom (str): Chromosome name, e.g., "chr1".
        start (int): 0-based start coordinate.
        end (int): End coordinate (non-inclusive).

    Returns:
        str: DNA sequence as a string (uppercase).
    """
    seq = genome[chrom][start:end].seq
    seq = str(seq).upper()
    return   seq, convert_seq(seq)



def get_clip_encoding(chrom, start,end):
  top_patterns = extract_top_binary_patterns(binary_df, chrom=chrom, start=start, end=end)
  seq_and_hot = get_sequence_from_fasta(chrom, start, end)
  top_out_process = np.array([[int(x) for x in x[0] ] for x in top_patterns])
  vector_amounts = np.array([x[1] for x in top_patterns])
  return seq_and_hot[0], seq_and_hot[1], top_out_process, vector_amounts


seq, seq_h_out, top_out,top_amount = get_clip_encoding('chr1', 100_000_000, 100_000_032)





# =============================================================================
# 1. Load promoter coordinates from BED file
# =============================================================================
promoter_bed = 'promoter_view_test.bed'  # replace with actual BED path
bed_df = pd.read_csv(
    promoter_bed,
    sep='\t',
    header=None,
    usecols=[0, 1, 2],
    names=['chrom', 'start', 'end']
)


LOAD_DATASET = False
# =============================================================================
# 2. Either load or sample n_samples random intervals ±25 kb of promoter start
# =============================================================================
if LOAD_DATASET:
    # Load pre-generated arrays
    seq_h_dataset      = np.load('seq_h_dataset.npy')
    top_out_dataset    = np.load('top_out_dataset.npy')
    top_amount_dataset = np.load('top_amount_dataset.npy')
else:
    # Initialize arrays to store generated samples
    seq_h_dataset      = np.zeros((n_samples, 5, 32),  dtype=np.float32)
    top_out_dataset    = np.zeros((n_samples, 10, 100), dtype=np.float32)
    top_amount_dataset = np.zeros((n_samples, 10),     dtype=np.float32)

    random.seed(42)
    i = 0
    pbar = tqdm(total=n_samples, desc="Generating samples")
    while i < n_samples:
        idx = random.randrange(len(bed_df))
        chrom = bed_df.loc[idx, 'chrom']
        prom_start = int(bed_df.loc[idx, 'start'])

        ext_start = max(0, prom_start - 25000)
        ext_end   = prom_start + 25000

        if ext_end - ext_start < 32:
            continue

        pos = random.randint(ext_start, ext_end - 32)
        _, seq_h_out, top_out, top_amount = get_clip_encoding(chrom, pos, pos + 32)

        if seq_h_out.shape != (5, 32) or top_out.shape != (10, 100) or top_amount.shape != (10,):
            continue

        seq_h_dataset[i]      = seq_h_out
        top_out_dataset[i]    = top_out
        top_amount_dataset[i] = top_amount
        i += 1
        pbar.update(1)
    pbar.close()

np.save('seq_h_dataset.npy', seq_h_dataset)
np.save('top_out_dataset.npy', top_out_dataset)
np.save('top_amount_dataset.npy', top_amount_dataset)
