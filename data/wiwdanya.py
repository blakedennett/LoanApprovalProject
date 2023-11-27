

# print("wiwdanya")

# x = 948576

# print(x)

# y = input("Enter a number: ")

# # y = '345'

# y = int(y) + 1

# print(y)


# x = [1471, 6171]

# for num in x:
#     print(num % 10)

import time
start_time = time.time()


X = 'jklcharacterwertyuicvbnmkjytrfvbnjuytrfvbnj'
X = 'characterfghjkltyuio'

# def L(i, j):

#     if i == j:

#         return 1
    
#     if X[i] == X[j]:
#         if i + 1 == j:
#             return 2
        
#         else:
#             return 2 + L(i + 1, j - 1)
        
#     else:

#         return max(L(i + 1, j), L(i, j - 1))
    
# result = L(0, len(X) - 1)

# print("Length of the longest palindrome:", result)

# print(time.time() - start_time)


# for item in completed:
#     print(item)

# 2716
# 40744
# 40035



import time
start_time = time.time()


memory_dict = {}

X = 'character'

def L(i, j):
    if i == j:
        return 1
    
    if (i, j) in memory_dict:
        return memory_dict[(i, j)]

    if X[i] == X[j]:
        if i + 1 == j:
            result = 2
        else:
            result = 2 + L(i + 1, j - 1)
    else:
        result = max(L(i + 1, j), L(i, j - 1))

    memory_dict[(i, j)] = result
    return result

result = L(0, len(X) - 1)

print("Length of the longest palindrome:", result)

print(time.time() - start_time)