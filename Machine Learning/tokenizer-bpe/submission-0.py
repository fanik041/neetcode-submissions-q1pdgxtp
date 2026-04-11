from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        tokens = list(corpus)

        merges = []

        # 2. For each merge step:
        for _ in range(num_merges):

            # a. Count frequency of all adjacent token pairs
            pair_counts = Counter()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            # b. Find the most frequent pair (break ties lexicographically)
            max_freq = max(pair_counts.values())
            candidates = [pair for pair, freq in pair_counts.items() if freq == max_freq]
            best_pair = min(candidates)  # lexicographic tie-break

            merges.append([best_pair[0], best_pair[1]])

            # c. Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])  # merge
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        # 3. Return the list of merges performed
        return merges