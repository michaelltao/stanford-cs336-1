from collections import defaultdict

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
    max_seq = (("", ""), 0)
    seq_counts = {}
    for seq, count in values.items():
        for i in range(len(seq) - 1):
            s = (seq[i], seq[i+1])
            seq_counts[s] = seq_counts.get(s, 0) + count
            if seq_counts[s] == max_seq[1]:
                new_seq = max([max_seq[0], s])
                max_seq = (new_seq, max_seq[1])
            elif seq_counts[s] > max_seq[1]:
                max_seq = (s, seq_counts[s])

    new_seqs = {}
    for seq, count in values.items():
        new_key = []
        i = 0
        while i < len(seq) - 1:
            s = (seq[i], seq[i+1])
            if s == max_seq[0]:
                new_char = f"{seq[i]}{seq[i+1]}"
                new_key.append(new_char)
                merges.add(" ".join([seq[i], seq[i+1]]))
                i += 2  # Jump the next character because it's now part of this merge
            else:
                # If no match, you might want to add the single char to new_key
                new_key.append(seq[i]) 
                if i+1 == len(seq) - 1:
                    new_key.append(seq[i+1])
                i += 1
        new_seqs[tuple(new_key)] = count

    values = new_seqs

    print(f"iteration... {iteration}")
    print(new_seqs)

print(merges)

