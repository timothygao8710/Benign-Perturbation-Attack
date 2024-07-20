from helper import *
import random

letters = 'abcdefghijklmnopqrstuvwxyz'

# get a random edit 1 edit distance away
def get_random_edit(word):
    C = random.choice(range(4))
    if len(word) <= 1:
        C = 3
        
    res='TEMP'
    if C == 0: # deletion
        i = random.choice(range(len(word)))
        res = word[:i] + word[i+1:]
    elif C == 1: # transpose:
        i = random.choice(range(len(word)-1))
        res = word[:i] + word[i+1] + word[i] + word[i+2:]
    elif C == 2: # replace:
        i = random.choice(range(len(word)))
        j = random.choice(letters)
        res = word[:i] + j + word[i+1:]
    else: # insert
        i = random.choice(range(len(word) + 1))
        j = random.choice(letters)
        res = word[:i] + j + word[i:]
    return res
    
print(get_random_edit('hello'))

def edits1(word):
    "All edits that are one edit away from `word`."    
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def get_all_edits(word, n=1):
    if n == 1:
        return edits1(word)
    
    last = get_all_edits(word, n-1)

    res = set()
    for i in last:
        cur = edits1(i)
        for j in cur:
            res.add(j)
    return res

# Sample word from starting word, n edit distance away
# def sample_one(word, n):
#     cur = select_random_subset(edits1(word), 1)
    
#     if n == 1:
#         return cur[0]
    
#     return sample_one(cur, n-1)

# def sample_edits(word, n, n_samples=1000):
#     level = select_random_subset(edits1(word), n_samples)
    
#     for _ in range(n-1): # ensure width of tree is bounded --> makes sampling linear instead of exponential with respect to n
#         nxt_level = []
#         for j in level:
#             nxt_level.append(select_random_subset(edits1(j), 1)[0])
#         level = nxt_level
            
#     return level

def sample_edits(word, n, n_samples=1000):
    res = []
    for i in range(n_samples):
        cur = word
        for j in range(n):
            cur = get_random_edit(cur)
        res.append(cur)
            
    return res

if __name__ == "__main__":
    print(sample_edits("hello", 1, 10))

    print(sample_edits("hello", 2, 10))

    print(sample_edits("hello", 3, 10))

    print(sample_edits("hello", 4, 10))

    print(sample_edits("hello", 1000, 10))
