from tqdm import tqdm

def count_sequences(input_fasta):
    """Counts number of sequences in a FASTA file (streaming)."""
    count = 0
    with open(input_fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def split_fasta(
    input_fasta,
    train_size,
    sampling_size,
    train_out,
    sample_prefix
):
    # Count sequences first
    total_sequences = count_sequences(input_fasta)
    print(f"Total sequences found: {total_sequences:,}")

    seq_count = 0
    sample_file_idx = 1
    writing_train = True

    train_file = open(train_out, "w")
    sample_file = None

    with open(input_fasta, "r") as f, tqdm(total=total_sequences, desc="Splitting FASTA") as pbar:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline()

            # Write to train set
            if writing_train:
                train_file.write(header)
                train_file.write(seq)
                seq_count += 1

                if seq_count >= train_size:
                    writing_train = False
                    train_file.close()
                    seq_count = 0
                    sample_file = open(f"{sample_prefix}_{sample_file_idx}.fasta", "w")

            # Write to sampling files
            else:
                sample_file.write(header)
                sample_file.write(seq)
                seq_count += 1

                if seq_count >= sampling_size:
                    sample_file.close()
                    sample_file_idx += 1
                    seq_count = 0
                    sample_file = open(f"{sample_prefix}_{sample_file_idx}.fasta", "w")

            pbar.update(1)

    # Close any open file handles
    if writing_train:
        train_file.close()
    else:
        if sample_file:
            sample_file.close()

    print("Splitting complete.")
