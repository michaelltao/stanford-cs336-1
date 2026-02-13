import collections


input_text = '''low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
'''

# split text by whitespace
pretokenized_text = input_text.split()

# count values
values = {}
for tok in pretokenized_text:
    seq = tuple(list(tok))
    values[seq] = values.get(seq, 0) + 1

# merge values
merges = set()

NUM_MERGES = 12

for iteration in range(NUM_MERGES):
    # 1. Count pairs
    seq_counts = collections.defaultdict(int)
    for seq, count in values.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            seq_counts[pair] += count

    if not seq_counts:
        break

    # 2. Find the best pair (max frequency)
    # Using max() with a key is cleaner than your manual if/else block
    best_pair = max(seq_counts, key=lambda x: (seq_counts[x], x))
    
    # 3. Create the new sequence mappings
    new_seqs = {}
    for seq, count in values.items():
        new_key = []
        i = 0
        while i < len(seq):
            # Check if we can even form a pair
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                new_key.append(seq[i] + seq[i+1]) # Merge!
                merges.add(" ".join(best_pair))
                i += 2
            else:
                new_key.append(seq[i]) # Keep original
                i += 1
        
        new_seqs[tuple(new_key)] = count

    values = new_seqs
    print(f"Iteration {iteration}: Merged {best_pair}")