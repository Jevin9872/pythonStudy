n = input()
list = input().split()
sum = 0
for i in range(int(n)):
    sum += int(list[i])
mean = sum/int(n)
# print(mean)
std = 0
for i in range(int(n)):
    std += (int(list[i]) - mean)**2
std = std/int(n)
# print(std)
for i in range(int(n)):
    print((int(list[i])-mean)/(std**0.5))
