import re

TANWEEN_FATEH = "\u064b"
TANWEEN_DAM   = "\u064c"
TANWEEN_KASER = "\u064d"
FATHA         = "\u064e"
DAMA          = "\u064f"
KASRA         = "\u0650"
SHADDAH       = "\u0651"
SUKUN         = "\u0652"
MADD          = "\u0653"


MADD2         = "\u06e4"
KENJAR_ALEF   = "\u0670"

ALL_HARAKAT   = FATHA+DAMA+KASRA+TANWEEN_FATEH+TANWEEN_DAM+TANWEEN_KASER+SHADDAH+SUKUN+MADD+MADD2

HAMZA = '\u0621'
ALEF  = '\u0627'
ALEF_UNDER_HAMZA = '\u0625'
ALEF_UPPER_HAMZA = '\u0623'
ALEF_MADD = '\u0622'

WAW = '\u0648'
WAW_HAMZA = '\u0624'
YAA = '\u064a'
SHORT_ALEF = '\u0649'
SHORT_ALEF_HAMZA = '\u0626'

HAA = '\u0647'
TIED_TAA = '\u0629'

ARABIC_LETTERS     = """\u0628\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\
\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0641\u0642\u0643\u0644\u0645\u0646"""

ARABIC_LETTERS += HAMZA+ALEF+ALEF_UNDER_HAMZA+ALEF_UPPER_HAMZA+ALEF_MADD
ARABIC_LETTERS += WAW+WAW_HAMZA+YAA+SHORT_ALEF+SHORT_ALEF_HAMZA+HAA+TIED_TAA+KENJAR_ALEF


NEW_NUMBERS = '0123456789'
OLD_NUMBERS = '\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669'


PUNCTUATION        = """\u0021\u0022\u0023\u0024\u0025\u0026\u0027\u0028\u0029\u002a\u002b\
\u002c\u002d\u002e\u002f\u003a\u003b\u003c\u003d\u003e\u003f\u0040\u005b\u005c\u005d\u005e\
\u005f\u0060\u007b\u007c\u007d\u007e\U0000201c\U0000201d\u061f\u060c\u061b\u00bb\u00ab"""


ALL_NUMBERS = NEW_NUMBERS + OLD_NUMBERS

QURANIC_SYMBOLS =  """\u06d6\u06d7\u06d8\u06d9\u06da\u06db\u06dc\u06e3\u06df\u06e0\u06e1\
\u06e2\u06e4\u06e5\u06e6\u06e7\u06e8\u06ea\u06eb\u06ec\u06ed\u06e9\u06dd\u06de\u08f0\u08f1\u08f2"""



#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



def removeNonAlphapit(text , exceptions='' ,alternative_text='', keep_space=True , keep_tashkeel=False) -> str:
    """
    ### replace all non alphapit in text with space
    * `exceptions` : specific character to keep in text - space by default
    """
    exceptions += (' ' if keep_space else "")
    return re.sub(rf'[^{ARABIC_LETTERS+exceptions+(ALL_HARAKAT if keep_tashkeel else "")}]', alternative_text , text)


def removeTashkeel(text , exceptions=''):
    """
    ### Remove harakat (decorations) 
    * `exceptions` : specific haraka to keep in text
    """
    global ALL_HARAKAT
    return re.sub(rf'[{"".join([i for i in ALL_HARAKAT if i not in exceptions])}]', "", text )



def removeTashkeel_(text , except_positions=[]):
    """
    ### Remove harakat (decorations) 
    `except_positions` : keep harakat on specific positions
    - Ex:
    * text = الْعَرَبِيَّةُ
    * remove_tashkeel_(text , [2,-1] ) => العَربيةُ
    * Keep the harakat on the third and last letter
    * index starts from 0
    """
    temp_harakat = []
    for word in text.translate(str.maketrans ( ARABIC_LETTERS , "|"*len(ARABIC_LETTERS) ) ).split():
        temp_harakat.append(word.split('|')[1:])
    words = []
    for index , word in enumerate(removeTashkeel(text).split()):
        word = list(word)
        for number in except_positions:
            try:
                    word[number] += temp_harakat[index][number]
                    temp_harakat[index][number] = ''
            except IndexError :pass
        
        words.append(''.join(word))
    return " ".join(words)



def removePunctuation(text , exceptions='', alternative_text=''):
    global PUNCTUATION
    """
    ### Remove punctuation
    * `exceptions` : specific special characters to keep in text
    """
    return re.sub(rf'[{"".join([i for i in PUNCTUATION if i not in exceptions])}]', alternative_text, text)





def keepAlphapitsAndNumbers(text, keep='' , alternative_text='' , keep_space=True , keep_tashkeel=False) -> str:
    """
    ### Keep only alphapits and numbers in the text
    `keep` : string of special characters to keep
    """
    keep += (' ' if keep_space else "")
    return re.sub(rf'[^{ARABIC_LETTERS+ALL_NUMBERS+keep+(ALL_HARAKAT if keep_tashkeel else "")}]', alternative_text , text)



def keepAlphapitsNumbersPunctuation(text, keep=''  , alternative_text='' , keep_space=True , keep_tashkeel=False) -> str:
    """
    ### Keep only alphapits and numbers and punctuation in the text
    `keep` : string of special characters to keep
    """
    keep += (' ' if keep_space else "")    
    return re.sub(rf"""[^{ARABIC_LETTERS+ALL_NUMBERS+PUNCTUATION+keep+(ALL_HARAKAT if keep_tashkeel else "")}]""", alternative_text , text)


def isArabicText(text , threshold=0.8) -> bool :
    """### Return Boolean , percent
    `threshold` : return True if percent greater than or equal threshold
    """
    len1 = len(removeNonAlphapit(text , keep_space=False))
    len2 = len(text.replace(" " ,''))-len(keepAlphapitsNumbersPunctuation(text,keep="ـ"+QURANIC_SYMBOLS,keep_space=False,keep_tashkeel=True))
    percent = len1/(len1+len2)
    return (True if percent >=threshold else False), percent


def isArabicWord(word, threshold=0.8 , withHarakat=True):
    percent = len(removeNonAlphapit(word ,exceptions="ـ", keep_space=False , keep_tashkeel=withHarakat))/len(word)
    return (True if percent >=threshold else False), percent


def removeNumbers(text , remove_old_numbers=True , remove_new_numbers=True , alternative_text=''):
    numbers = (OLD_NUMBERS if remove_old_numbers else "")+(NEW_NUMBERS if remove_new_numbers else "")
    clean_text = text
    for number in numbers:
        clean_text = clean_text.replace(number , alternative_text)
    return clean_text


def extractEnglishText(text):
    english_words = re.findall(r'[a-zA-Z]+(?:[-\'][a-zA-Z]+)*', text)
    return english_words  

def oldNumbersReplacment(text , reverse=False):
    classes = [OLD_NUMBERS , NEW_NUMBERS]
    for class1 , class2 in zip(classes[reverse] , classes[1-reverse] ) :
        text = text.replace(class1 , class2)
    return text

def kenjarAlefReplacment(text):
    return text.replace(KENJAR_ALEF , ALEF )


def extractNumbers(text):
    return re.findall(r'\d+', text)


def normalizeAlef(text):
    for  i in ALEF_UNDER_HAMZA+ALEF_UPPER_HAMZA+ALEF_MADD:
        text = text.replace(i, ALEF )
    return text


def normalizeHamza(text):
    for  i in ALEF_UNDER_HAMZA+ALEF_UPPER_HAMZA+SHORT_ALEF_HAMZA+WAW_HAMZA:
        text = text.replace(i,HAMZA)
    return text


def disassembleMadd(text):
    return text.replace( ALEF_MADD , ALEF_UPPER_HAMZA+ALEF )



if __name__ == "__main__":
    print(isArabicWord("الْـعَرَبِيَّةُ",withHarakat=True))