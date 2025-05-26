
path = r'D:\py\final_project\AI_model\data_collection_and_preprocessing\data_collection\Roots'

with open(path+r'\all_roots3.txt' , 'rb') as file :
    all_roots3 = set(file.read().splitlines())

with open(path+r'\all_roots4.txt' , 'rb') as file :
    all_roots4 = set(file.read().splitlines())

with open(path+r'\UnWanted_roots3.txt' , 'rb') as file :
    UnWanted_roots3 = set(file.read().splitlines())

with open(path+r'\UnWanted_roots4.txt' , 'rb') as file :
    UnWanted_roots4 = set(file.read().splitlines())

print(len(all_roots3)," - ",len(UnWanted_roots3) ," = " , end="")
all_roots3 -= UnWanted_roots3
print(len(all_roots3))
print(len(all_roots4)," - ",len(UnWanted_roots4) ," = " , end="")
all_roots4 -= UnWanted_roots4
print(len(all_roots4))


with open(path+r'\Roots3.txt' , 'wb') as file :
    file.writelines(sorted(map(lambda x : x+b'\n' ,all_roots3)))

with open(path+r'\Roots4.txt' , 'wb') as file :
    file.writelines(sorted(map(lambda x : x+b'\n' ,all_roots4)))