
import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:

    N = len(seqs)

    # Step 1: Determine target length
    if max_len is None:
        max_len = 0

        for i in range(N):
            if max_len < len(seqs[i]):
                max_len = len(seqs[i])

    # Step 2: Handle empty input
    if N == 0:
        return np.empty((0, 0), dtype=int)

    # Step 3: Pad or truncate each sequence
    result = []

    for i in range(N):
        seq = seqs[i]

        # Truncate if sequence is too long
        seq = seq[:max_len]

        # Calculate required padding
        padding_needed = max_len - len(seq)

        # Create padding
        padding = [pad_value] * padding_needed

        # Combine original sequence and padding
        new_seq = seq + padding

        result.append(new_seq)

    # Step 4: Convert to NumPy array
    return np.array(result, dtype=int)
