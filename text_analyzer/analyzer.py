import pickle
import os
import sys

__version__ = 1.0

folder_path = os.path.dirname(__file__)

sys.path.append(folder_path)
sys.path.append(os.path.dirname(folder_path))

PUNCTUATION        = """\u0021\u0022\u0023\u0024\u0025\u0026\u0027\u0028\u0029\u002a\u002b\
\u002c\u002d\u002e\u002f\u003a\u003b\u003c\u003d\u003e\u003f\u0040\u005b\u005c\u005d\u005e\
\u005f\u0060\u007b\u007c\u007d\u007e\U0000201c\U0000201d\u061f\u060c\u061b\u00bb\u00ab"""

try :import filtering # type:ignore
except ImportError :raise ImportError("Faild to import 'filtering' , Make sure to install all requirement.")

try :import inference # type:ignore
except ImportError :raise ImportError("Faild to import 'inference' , Make sure to install all requirement.")

try :import match # type:ignore
except ImportError :raise ImportError("Faild to import 'match' Lib , Make sure to install all requirement.")
with open(folder_path+"/final_dict_triple.pickle" , 'rb')  as file :
    triple_word_weight_dict = pickle.load(file)

with open(folder_path+"/final_dict_quaternary.pickle" , 'rb')  as file :
    quaternary_word_weight_dict = pickle.load(file)

with open(folder_path+"/dialect_words_dict_triple.pickle" , 'rb')  as file :
    dialect_words_dict_triple = pickle.load(file)


all_stop_words = {}
with open(folder_path+r"\lemm\all_stop_words_dict.txt" , 'rb')  as file :
    basic_stop_words1 = {line[:line.find(":")]:line[line.find(":")+1:] for line in file.read().decode().splitlines()}

with open(folder_path+r"\lemm\non_lemmatized_stop_word_dict.txt" , 'rb')  as file :
    basic_stop_words2 = {line[:line.find(":")]:line[line.find(":")+1:] for line in file.read().decode().splitlines()}

with open(folder_path+r"\lemm\times_dict.txt" , 'rb')  as file :
    time_stop_words = {line[:line.find(":")]:line[line.find(":")+1:] for line in file.read().decode().splitlines()}

with open(folder_path+r"\lemm\money_dict.txt" , 'rb')  as file :
    money_stop_words = {line[:line.find(":")]:line[line.find(":")+1:] for line in file.read().decode().splitlines()}

with open(folder_path+r"\lemm\egypt_stop_words_dict.txt" , 'rb')  as file :
    dialect_stop_words = {line[:line.find(":")]:line[line.find(":")+1:] for line in file.read().decode().splitlines()}

with open(folder_path+r"\lemm\shami_stop_words_dict.txt" , 'rb')  as file :
    dialect_stop_words = dialect_stop_words | ({line[:line.find(":")]:line[line.find(":")+1:] for line in file.read().decode().splitlines()})

all_stop_words = basic_stop_words1 | basic_stop_words2 | time_stop_words | money_stop_words | dialect_stop_words


lemm_dict = {}
pos_dict = {}
with open(folder_path + r"\lemm\triple_weigths_lemm_and_pos.txt" , 'rb') as file :
    for line in file.read().decode().splitlines():
        line = line.replace("\t" , "")
        first_index = line.find(":")
        second_index = line.rfind(":")
        lemm_dict.update([[line[:first_index] , line[second_index+1:] ]])
        pos_dict. update([[line[:first_index] , line[first_index+1:second_index] ]])


def get_most_similar(weight , top=10 , matching_strength=3, task_type=None):
    if task_type == "lemm":
        result = []
        if lemm_dict.get(weight , None):
            return [(1. , lemm_dict.get(weight , None))]

        for w in lemm_dict.keys():
            result.append((match.weight_matching(weight  , w ,matching_strength) , w))
        return sorted(result)[-1:-top-1:-1]
    if task_type == "pos":
        result = []
        for w in pos_dict.keys():
            result.append((match.weight_matching(weight  , w ,matching_strength) , w))
        return sorted(result)[-1:-top-1:-1]



def get_label(mathes , task_type=None):
    result = dict()
    if task_type == "lemm":
        for index ,(rate , weight) in enumerate(mathes) :
            if lemm_dict[weight] not in result.keys():
                result.update([[lemm_dict[weight] , rate*(1+rate)]])
            else :
                result[lemm_dict[weight]]+=rate*(1+rate)
    if task_type == "pos":
        for index ,(rate , weight) in enumerate(mathes) :
            if pos_dict[weight] not in result.keys():
                result.update([[pos_dict[weight] , rate*(1+rate)]])
            else :
                result[pos_dict[weight]]+=rate*(1+rate)
            
    maxmim_score = 0
    predicted_weight = ""
    for key , value in result.items():
        if value>maxmim_score:
            maxmim_score = value
            predicted_weight=key
    return predicted_weight

def merge(root:list , weight:list  , weight_indices ):
    temp_weight = weight.copy()
    temp_weight[weight_indices[0]] = root[0]
    temp_weight[weight_indices[1]] = root[1]
    temp_weight[weight_indices[2]] = root[2]
    return temp_weight



def get_original_letters_index_weight(weight:list):
    indices = []
    weight_len = len(weight)
    
    length = (3 if ('\x00' in weight or '\x01' in weight) else 4)
    for index , letter in enumerate(weight[::-1]) :
        if len(indices) == length : return indices[::-1]
        if   letter == '\x01' :indices.append(weight_len-index-2)
        elif letter == '\x00' :indices.append(weight_len-index-2)
        else :
            if letter == 'ل' : indices.append(weight_len-index-1)
            if letter == 'ع' : indices.append(weight_len-index-1)
            if letter == 'ف' : indices.append(weight_len-index-1) ; return indices[::-1]
    return indices[::-1]




def get_root3(word:list , weight:list):
    temp_weight = weight[:]
    try :
        if temp_weight[weight.index('ف'):].count('ل') > 1 :temp_weight["".join(weight).rfind('ل')] = '0'
    except:pass
    root = []
    shift = temp_weight.count('\x01') + temp_weight.count('\x00')*2
    state = False
    for index , letter in list(enumerate(temp_weight))[::-1] :
        if state :state=False;continue
        if len(root) == 3 : return root[::-1]

        if   letter == '\x01' : root.append(temp_weight[index-1]);shift-=1
        elif letter == '\x00' : root.append(temp_weight[index-1]);shift-=2;state=True
        
        else :
            if letter == 'ل' :root.append(word[index-shift])
            if letter == 'ع' :root.append(word[index-shift])
            if letter == 'ف' :root.append(word[index-shift]) ; return root[::-1]
        
    return root[::-1]

def get_root4(word:list , weight:list):
    root = []    
    for index , letter in list(enumerate(weight))[::-1] :
        if len(root) == 4 : return root[::-1]
        else :
            if letter == 'ل' :root.append(word[index])
            if letter == 'ع' :root.append(word[index])
            if letter == 'ف' :root.append(word[index]) ; return root[::-1]
    return "".join(root[::-1])


def get_root (key , word , list ):

    if key == "3" :
        result = get_root3(word , list ) 
        return result if "\x00" not in result and "\x01" not in result  else None
    if key == "4" :
        result = get_root4(word , list ) 
        return result if "\x00" not in result and "\x01" not in result  else None


def get_real_weight(weight:list):
    real_weight = []
    base_weight = ['ف' , 'ع' , 'ل']
    
    counter = 0
    state = False
    for index , letter in list(enumerate(weight))[::-1] :
        if state :state=False ; continue
        
        if   letter == '\x01' :real_weight.append(base_weight[ 2 - counter]);state = True
        elif letter == '\x00' :real_weight.append(base_weight[ 2 - counter]);state = True
        else :
            if   letter == 'ل' : real_weight.append(letter);counter+=1
            elif letter == 'ع' : real_weight.append(letter);counter+=1
            elif letter == 'ف' : real_weight.append(letter);counter+=1
            else :real_weight.append(letter)

    return real_weight[::-1]


def get_real_weights(weights):
    resulte = []

    for weight in weights :
        if "\x00" not in weight and "\x01" not in weight :resulte.append("".join(weight))
        else :resulte.append("".join(get_real_weight(weight)))
    return resulte


def get_root_type(word , weight): # output :  3 OR 4 
    counter = 0
    if weight[2:].count('ل') > 1 :
        for  index , letter in enumerate(weight[2:]) :
            if letter == 'ل' and word[index+2] != 'ل' :
                counter+=1
            if counter > 1 :return "4"
    return "3"





class Word :
    def __init__ (self, word:str , weights:dict , stop_word=False):

        self.stop_word = stop_word

        self.root = self.prepare_roots(word ,weights ) if not stop_word else self.prepare_stop_word_roots(word)
        
        self.real_weight = {key:"".join(get_real_weights(weight)) for key , weight in weights.items() if weight != None}
        self.word = word
        self.deep_weight = {key:"".join(weight) for key , weight in weights.items() if weight != None}
        

    def prepare_roots(self, word , weights):
        roots = {}
        for key in weights.keys() :
            if weights[key] :
                roots[key] = "".join(get_root(key ,list(word) , weights[key]))
        return roots

    def prepare_stop_word_roots(self , word) :
        if all_stop_words.get(word , None):
            result = all_stop_words.get(word , None) 
        else :
            result = all_stop_words.get(filtering.normalizeAlef(word).replace("ة" , "ه") , None)
            
        return [result]

    def matching (self , word):
        weight_m = match.weight_matching(self.get_weight[0] , word.get_weight[0] , min(len(self.get_weight[0]) , len(word.get_weight[0]))-1 )
        root_m = 0.5 if self.get_root[0] == word.get_root[0] else 0
        return root_m + weight_m/2
        
         

    @property
    def get_root(self):
        if self.is_stop_word:return self.root
        return list(self.root.values())
        
    @property
    def get_weight(self):
        if self.is_stop_word:return None
        return list(self.real_weight.values())
    
    @property
    def get_deep_weight(self):return self.deep_weight
    @property
    def is_stop_word(self):return self.stop_word

    def __repr__(self):
        return self.word
    @property
    def get_word_lemm(self):
        word_lemm = {}
        if self.is_stop_word : return self.root
        for key , weight in self.get_weight_lemm.items():
            if key=="3":
                try:
                    word_list = merge(list(self.root[key]) , list(weight),get_original_letters_index_weight(list(weight)))
                    word_lemm.update([[key,"".join(word_list)]])
                except:return list(self.root.values())
            elif key=='4':
                word_lemm.update([[key,self.word]])
        return list(word_lemm.values())
            
    @property
    def get_weight_lemm(self):
        if self.is_stop_word :return None
        
        weight_lemm_dict = {}
        for key , weight in self.real_weight.items():
            if key == "3":                
                weight_lemm_dict.update([[key,get_label(get_most_similar(weight , 10 , 3 if len(self.real_weight[key])<6 else 4 , task_type="lemm"), task_type="lemm")]])
            elif key == "4":
                weight_lemm_dict.update([[ key  , weight  ]])
        return weight_lemm_dict

            
    @property
    def get_pos_tag(self):
        if self.is_stop_word :return ['حرف']
        weight_pos_dict = {}
        for key , weight in self.real_weight.items():
            if key == "3":                
                weight_pos_dict.update([[key,get_label(get_most_similar(weight , 10 , 3 if len(self.real_weight[key])<6 else 4 , task_type="pos"), task_type="pos")]])
            elif key == "4":
                weight_pos_dict.update([[ key  , weight  ]])
        return list(weight_pos_dict.values())
    @property
    def is_punctuation (self):return False
    
    

class NonWord :
    def __init__(self , word , is_punctuation=False):
        self.word = word
        self.punctuation = is_punctuation
    @property
    def get_root(self):return None
    @property
    def get_weight(self):return None
    @property
    def get_deep_weight(self):return None
    @property
    def is_stop_word(self):return None
    @property
    def get_word_lemm(self):return None
    @property
    def get_weight_lemm(self):return None
    @property
    def get_pos_tag(self):return None
    @property
    def is_punctuation (self):return self.punctuation
    
    

    def __repr__(self):return self.word
    

def clean_text(word:str) -> str :
    word = filtering.removeTashkeel(word)
    word = filtering.kenjarAlefReplacment(word)
    return word



def analyze (words:list[str]) -> list[Word,NonWord] :
    words_obj_list = []
    for w in words :
        try:
            w = filtering.removeTashkeel(w)

            weights:dict[str,list] = {"3": None , '4': None }
            if w in all_stop_words.keys() :words_obj_list.append(Word(w , weights , True)) ; continue
            if filtering.normalizeAlef(w).replace("ة" , "ه") in all_stop_words.keys() :words_obj_list.append(Word(w , weights , True)) ; continue


            
            if all(i in PUNCTUATION for i in w ) : words_obj_list.append(NonWord(w , True));continue
            if not filtering.isArabicWord(w , threshold=1. )[0] :words_obj_list.append(NonWord(w));continue
            

            weights["3"] = dialect_words_dict_triple.get(w , None)
            
            if not weights["3"]:weights["3"] = dialect_words_dict_triple.get(filtering.normalizeHamza(w) , None)

            
            if not weights["3"]:
                weights["3"] = triple_word_weight_dict.get(w , None)
            if not weights["3"]:
                expected_weight = inference.extract_weight(folder_path , w)
                if expected_weight :
                    weights[get_root_type(w , expected_weight)] = expected_weight
                    print(f"[{w}] model predicted")
                else :
                    weights["3"] =  None
            
            

            if w in quaternary_word_weight_dict.keys() : weights['4'] = list(quaternary_word_weight_dict[w])

            weights['4'] = quaternary_word_weight_dict.get(w , None)
            
            if list(weights.values()).count(None) == 2 : words_obj_list.append(NonWord(w))
            else :words_obj_list.append(Word(clean_text(w) , weights))
            
        except Exception as e :print(f"[-] Error [{e}] | Unknown word {w}");words_obj_list.append(NonWord(w))

    return words_obj_list





if __name__ == "__main__":
    text = """رأيت الاصدقاء يجلسون هناك"""
    words = analyze(text.split())
    print("word    word.get_root    word.get_weight    word.get_weight_lemm   word.get_pos_tag    word.get_deep_weight     word.is_stop_word     type(word) ")
    for word in words :
        print("="*10)
        print(word , word.get_root, word.get_weight, word.get_weight_lemm,word.get_pos_tag, word.get_deep_weight , word.is_stop_word , type(word) , sep='|')
    word1 = analyze(['فارس'])[0]
    print(word1 , word1.get_root , word1.get_pos_tag )
    word2 = analyze(['فرسان'])[0]
    print(word2 , word2.get_root , word2.get_pos_tag )
    print(word1.matching(word2))



