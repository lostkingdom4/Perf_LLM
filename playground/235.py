def main():
    N, *A = list(map(int, open(0).read().split()))
    for i in range(N):
        A[i] -= i + 1
    A.sort()
    b = A[N // 2]
    print((sum([abs(A[i] - b) for i in range(N)])))
    return


main()