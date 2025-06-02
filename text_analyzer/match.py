



def list_power( value:str , max_len=3):
    new_list = []
    max_len = max_len if max_len <= len(value) else len(value)
    
    for i in range(0 , len(value)-max_len+1):
        for max_length in range(max_len , 1 , -1) :
            if max_length > len(value) :continue
            new_list.append(value[i:i+max_length])

    for index in range(i+1,len(value)-1):
        for index2 in range(len(value) - index , 1 , -1) :
            new_list.append(value[index :index+index2])

    new_list.extend(list(value))
    return new_list


def intersection(list1, list2):
    return [item for item in list1 if item in list2]

def difference(list1, list2):
    return [item for item in list1 if item not in list2]

def union(list1 , list2) :
    new_list = []
    for pair in zip(list1 , list2):
        for value in pair:
            if value not in new_list : new_list.append(value)
    return new_list

def one_hot_intersection(sub_list , full_list) :
    new_list = []
    for value in full_list :
        if value in sub_list :
            new_list.append(len(value))
        else :
            new_list.append(0)
    return new_list


def calculate_similarity(l:list):
    sum = 0
    counter = len(l)
    for value in l :
        sum += 2**(value-1) #+ ((counter/len(l))*counter)
        counter -= (1 if counter > (len(l)/2) else 0.5)
    return sum

def simple_str_match(s1:str , s2:str) :
    return sum(c1 == c2 for c1, c2 in zip(s1, s2))


def calculate_sub_similarity(set1 , set2):
    sum = 0
    for item1 in set1 :
        if len(item1) == 1 :continue
        for item2 in set2 :
            if len(item2) == 1 :continue
            if len(item1) == len(item2):
                sum += simple_str_match(str(item1) , str(item2))
    return sum


def weight_matching(weight1 , weight2 , matching_strength=3):
    list1 = list_power(weight1,matching_strength)
    list2 = list_power(weight2,matching_strength)
    
    full_list = union(list1 , list2)
    intersecte_list1_list2 = intersection(list1,list2)
    
    similarity = calculate_similarity(one_hot_intersection(intersecte_list1_list2 , full_list))
    sub_similarity = calculate_sub_similarity(difference(list1 , list2 ),difference(list2 , list1 ) )
    

    
    base = calculate_similarity(one_hot_intersection(list1 if len(list1)>=len(list2) else list2 , full_list) )
    return ( similarity+ (sub_similarity)/3)/(base)




weight1 = "يفعل"
weight2 = "تفعل"
weight2 = "مفعول"
if __name__ == "__main__":
    print("resulte : ",weight_matching(weight1, weight2  ,5))
