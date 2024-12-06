arr= [1,2,4,5,6,2,7,2]
print(len(arr))
def index_arr(arr,target):
    result_dict={}

    new= arr.copy()
    new1 ,new2 = new[:len(new)//2] , new[len(new)//2: ]
    for idx,(i,j) in enumerate(zip(new1,new2)):
        if target == i :
            result_dict[idx]=target 
        elif target == j:
            result_dict[len(new1) + idx] = target
    result_dict=sorted(result_dict)
    while len(result_dict) != 1:
        result_dict.pop()
    return result_dict[0]
        
print(index_arr(arr,2))