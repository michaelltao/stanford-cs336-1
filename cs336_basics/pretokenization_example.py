import os
import regex as re
import heapq
from typing import BinaryIO
from collections import Counter, defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def merge_dict(data: dict[tuple[str, ...], int], num_merges: int):
    # pair -> []word mapping
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

    
    for iter in range(num_merges):
        while heap:
            neg_cnt, top_pair = heapq.heappop(heap)
            if seq_counter[top_pair] == -neg_cnt:
                break
        else: break

        affected_seqs = list(inverted_idx[top_pair])

        for old_seq in affected_seqs:
            print(old_seq)
            count = data.pop(old_seq)
            for i in range(len(old_seq) - 1):
                p = (old_seq[i], old_seq[i+1])
                print('p', p)
                seq_counter[p] -= count
                print(inverted_idx[p])
                inverted_idx[p].remove(old_seq)

            x = 0
            new_seq = []

            newidk = defaultdict(int)
            while x < len(old_seq):
                if x < len(old_seq) - 1 and old_seq[x:x+2] == top_pair:
                    joined = ("".join(old_seq[x:x+2]))
                    new_seq.append(joined)

                    x += 2
                else:
                    new_seq.append(old_seq[x])
                    x += 1
            
            new_seq = tuple(new_seq)
            data[new_seq] = count
            for i in range(len(new_seq) - 1):
                p = (new_seq[i], new_seq[i+1])
                seq_counter[p] -= count 
                inverted_idx[p].add(new_seq) 

                heapq.heappush(heap, (-seq_counter[p], p))
        
    return


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

## Usage
with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        pretokenized_text = re.findall(PAT, chunk)
        text = [tuple(list(word)) for word in pretokenized_text]
        counted = Counter(text)
        merge_dict(counted, 1)
        break
