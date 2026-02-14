from collections import Counter, defaultdict

import regex as re
import heapq

from cs336_basics.pretokenization_example import find_chunk_boundaries

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> (dict[int, bytes], list[tuple[bytes, bytes]]):

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    special_tokens_bytes = [tok.encode("utf-8") for tok in special_tokens]
    split_tokens = special_tokens_bytes or [b"<|endoftext|>"]
    boundary_token = split_tokens[0]

    with open(input_path, "rb") as f:
        num_processes = 4

        boundaries = find_chunk_boundaries(f, num_processes, boundary_token)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        final_toks = Counter()
        split_pat = b"|".join(re.escape(tok) for tok in sorted(split_tokens, key=len, reverse=True))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start)
            split_texts = re.split(split_pat, chunk)

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            for split_bytes in split_texts:
                text = split_bytes.decode("utf-8", errors="ignore")
                pretokenized_text = re.findall(PAT, text)
                toks = [tuple(list(word)) for word in pretokenized_text]
                final_toks |= Counter(toks)

        final_vocab, merges = merge_dict(
            final_toks,
            vocab_size=vocab_size,
            special_tokens=special_tokens_bytes,
        )

    return final_vocab, merges


def merge_dict(
    data: dict[tuple[str, ...], int], 
    vocab_size: int,
    special_tokens: list[bytes]
) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []

    inverted_idx = defaultdict(set)
    
    # pair -> count mapping
    seq_counter = defaultdict(int)
    for seq, count in data.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            seq_counter[pair] += count
            inverted_idx[pair].add(seq)

    # Build max-heap: (negative_count, pair)
    heap = [(-cnt, pair) for pair, cnt in seq_counter.items()]
    heapq.heapify(heap)

    
    while True:
        while heap:
            neg_cnt, top_pair = heapq.heappop(heap)
            if seq_counter[top_pair] == -neg_cnt:
                break
        else: break

        top_pair_bytes = (top_pair[0].encode("utf-8"), top_pair[1].encode("utf-8"))
        merged_token = top_pair_bytes[0] + top_pair_bytes[1]
        merges.append(top_pair_bytes)
        vocab[next_id] = merged_token
        next_id += 1

        if len(vocab) >= vocab_size:
            return vocab, merges

        affected_seqs = list(inverted_idx[top_pair])

        for old_seq in affected_seqs:
            count = data.pop(old_seq)
            for i in range(len(old_seq) - 1):
                p = (old_seq[i], old_seq[i+1])
                seq_counter[p] -= count
                inverted_idx[p].discard(old_seq)

            x = 0
            new_seq = []

            while x < len(old_seq):
                if x < len(old_seq) - 1 and old_seq[x:x+2] == top_pair:
                    joined = "".join(old_seq[x:x+2])
                    new_seq.append(joined)

                    x += 2
                else:
                    new_seq.append(old_seq[x])
                    x += 1
            
            new_seq = tuple(new_seq)
            data[new_seq] = count
            for i in range(len(new_seq) - 1):
                p = (new_seq[i], new_seq[i+1])
                seq_counter[p] += count 
                inverted_idx[p].add(new_seq) 

                heapq.heappush(heap, (-seq_counter[p], p))

    return vocab, merges
