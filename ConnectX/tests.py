import numpy as np

arr = np.arange(6).reshape(3,2)
for x in range(2):
    print(0 in arr[x])

print(arr)
print(arr[0,0])

