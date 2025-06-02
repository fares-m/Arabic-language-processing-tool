import re
from collections import Counter
import os , sys

folder_path = os.path.dirname(__file__)
sys.path.append(folder_path)

from filtering import PUNCTUATION # type:ignore


pattern1 = '\u0621-\u063a\u0641-\u064a\u064b-\u0652' # arabic letters and harakat
pattern2 = '\u0660-\u0669' # arabic numbers

WordTokenizerPattern = re.compile(r"\b(?:[%s]{1,2}\.)+[%s]{1,2}\b|[%s]+|\d+|[%s]+|[%s]+|[^\s\d%s]+"%(pattern1, pattern1, pattern1,pattern2, PUNCTUATION,PUNCTUATION))


def WordTokenization(text):
    return re.findall(WordTokenizerPattern, text)


def split_text(text):    
    keywords = ["و", "ثم", "أو", "بل", "لا", "لكن", "حتى", "أي", "لأن", "فإن", 
               "إذن", "بسبب", "حيث", "عندما", "حين", "قبل", "منذ", "كلما", 
               "بينما", "إذا", "لو", "حيثما", "إن", "أكثر من", "مثل", "كما", 
               "غير أن", "مع ذلك", "بالرغم من", "ايضا", "كذلك", "خصوصا", "خاصة"]
        
    punctuation_marks = [".", "!", "؟", "؛", "...", ',', '،']
    
    sentences = []
    current_sentence = ""
    
    i = 0
    while i < len(text):
        char = text[i]
        current_sentence += char
        
        if char in punctuation_marks:            
            if char == "." and i+2 < len(text) and text[i+1] == "." and text[i+2] == ".":
                current_sentence += ".."
                i += 2
            sentences.append(current_sentence.strip())
            current_sentence = ""
            i += 1
            continue
            
            
        if char == " ":
            next_word = ""
            j = i + 1
            while j < len(text) and text[j] != " ":
                next_word += text[j]
                j += 1
                
            if next_word in keywords and len(current_sentence) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = next_word + " "
                i = j - 1
                
        i += 1
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences




class Count_Dictionary : 
    def __init__(self , text) -> None :
        self.__dictionary = Counter(text)
        

    def __len__(self):
        return len(self.__dictionary.keys())
    
    def __getitem__(self , key):return self.__dictionary[key]


    @property
    def values(self):return list(self.__dictionary.values())
    @property
    def keys(self):return list(self.__dictionary.keys())
    
    def get_weight(self,key):
        """ More rarely word more valuable word. """
        return 1-(self.__dictionary[key]/self.__len__())

    def getValueByKey(self,key):return self.__dictionary[key]

    def getKeysByValue(self,value): return [word for word, count in self.__dictionary.items() if count == value]

    def word_weight_in_doc(self , doc_text , word ):
        """
        Word importance in a specific document.
        """
        if word not in self.__dictionary:self.add_text(doc_text)
        try:return doc_text.split().count(word) * self.__dictionary[word]
        except KeyError :return 0

    def words_count(self):
        return self.__len__()
    
    
    def expand_dictionary(self , text):
        text = text.split()
        self.__dictionary+=Counter(text)

    def dectionary(self):
        return self.__dictionary
