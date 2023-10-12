n = int(eval(input()))
s = 10**11
for i in range(1, int(n**0.5) + 1):
    if n % i == 0:
        s = min(s, max(i, n // i))
print((len(str(s))))
