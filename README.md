# Arabic-language-processing-tool
Arabic lang processing tool , for preprocessing Arabic text 
Verson 1.0

### Filtering 
contain a number of functions to preprocesse arabic text such as :
1. removeNonAlphapit
2. removeTashkeel
3. isArabicText
4. normalizeAlef
5. normalizeHamza
6. And so on .

* The defferance between removeTashkeel and removeTashkeel_ 
that removeTashkeel_ could keep a haraka on a spicfic letter
for example : 
except_positions = [1 , -1] , this mean keep the second haraka and the last one.

---

* Another defferance between isArabicWord and isArabicText

“IsArabicWord” function use to determine whether the word has only Arabic letters and only harakat if “withHarakat=True” .
While “isArabicText” function determines whether the whole text is Arabic, means that punctuation will not reduce the pure rate of the text unlike “isArabicWord”.

### Tokenization

* containing tow functions :
1. word tokenizer
2. sentenc tokenizer







