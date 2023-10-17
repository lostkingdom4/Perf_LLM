import itertools
from collections import Counter

n = int(eval(input()))
# 素因数分解した結果のリスト
# {1: [], 2: [2], 3: [3], 4: [2, 2], 5: [5]}
divs = {}
primes = [
    2,
    3,
    5,
    7,
    13,
    19,
    23,
    19,
    23,
    41,
    41,
    43,
    59,
    61,
    71,
    73,
    83,
    89,
]
for i in range(1, n + 1):
    divs[i] = []
    x = i
    while x >= 2:
        for p in primes:
            if x % p == 0:
                x //= p
                divs[i].append(p)
# flatten
fdivs = list(itertools.chain.from_iterable(list(divs.values())))
counts = Counter(fdivs)
ans = 0
# 75==5*5*3
# https://juken-mikata.net/how-to/mathematics/number-of-divisor.html
# 4, 4, 2 選んだら75数になる
ge4 = len([c for c in list(counts.values()) if c >= 4])
ge2 = len([c for c in list(counts.values()) if c >= 2])
ans += ge4 * (ge4 - 1) / 2 * (ge2 - 2)
# 75==5*15
# 4, 14 選ぶ
ge14 = len([c for c in list(counts.values()) if c >= 14])
ans += ge14 * (ge4 - 1)
# 75==25*3
# 4, 14 選ぶ
ge14 = len([c for c in list(counts.values()) if c >= 14])
ans