n = int(input())
num_str = input()
num_list = [int(char) for char in num_str]
print(num_list)
min_result = int(num_str, 2)
for i in range(n):
    if num_list[i] == 0:
        break
    meet_zero = False
    for j in range(i+1,n):
        if num_list[j] == 0:
            meet_zero = True
        if meet_zero and num_list[j]==1:
            break
    temp_num_list = num_list[:]
    temp_num_list[i:j] = list(reversed(temp_num_list[i:j]))
    temp_num_list_str = [str(num) for num in temp_num_list]
    
    temp_result_str = ''.join(temp_num_list_str)
    temp_result = int(temp_result_str,2)
    if temp_result < min_result:
        min_result = temp_result
        result = temp_result_str

print(result)
