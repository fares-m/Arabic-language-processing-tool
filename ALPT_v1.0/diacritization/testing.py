import numpy as np
import pickle as pkl
from tensorflow.keras.models import load_model # type:ignore
from tensorflow.keras.utils import to_categorical # type:ignore
import re


path = __file__[:__file__.rfind("\\")]


# Load mappings and character sets
with open(path+'/ARABIC_LETTERS_LIST.pickle', 'rb') as file:
    a_l_l = pkl.load(file)

with open(path+'/DIACRITICS_LIST.pickle', 'rb') as file:
    d_l = pkl.load(file)

with open(path+'/RNN_CLASSES_MAPPING.pickle', 'rb') as file:
    class_m = pkl.load(file)

with open(path+'/RNN_REV_CLASSES_MAPPING.pickle', 'rb') as file:
    r_c_m = pkl.load(file)

with open(path+'/RNN_SMALL_CHARACTERS_MAPPING.pickle', 'rb') as file:
    char_m = pkl.load(file)

# Utility functions
def remove_diacritics(data):
    return "".join([char for char in data if char not in d_l])

def clean_text(text):
    return re.sub(r"[-()\"#/@;:_<>{}`﴾﴿ے+=~&|.!?,a-zA-Z0-9٠-٩…''•]", "", text)

def map_data(data_raw):
    X = []
    Y = []

    for line in data_raw:
        x = [char_m['<SOS>']]
        y = [class_m['<SOS>']]

        for idx, char in enumerate(line):
            if char in d_l:
                continue
            x.append(char_m.get(char, char_m['<UNK>']))  # handle unknown chars

            if char not in a_l_l:
                y.append(class_m[''])
            else:
                char_diac = ''
                if idx + 1 < len(line) and line[idx + 1] in d_l:
                    char_diac = line[idx + 1]
                    if idx + 2 < len(line) and line[idx + 2] in d_l and char_diac + line[idx + 2] in class_m:
                        char_diac += line[idx + 2]
                    elif idx + 2 < len(line) and line[idx + 2] in d_l and line[idx + 2] + char_diac in class_m:
                        char_diac = line[idx + 2] + char_diac
                y.append(class_m.get(char_diac, class_m['']))

        x.append(char_m['<EOS>'])
        y.append(class_m['<EOS>'])
        y = to_categorical(y, len(class_m))

        X.append(x)
        Y.append(y)

    return X, Y

# Prediction function
def predict_with_confidence(line, model, confidence_threshold=0.5):
    X, _ = map_data([line])
    X = np.asarray(X)
    predictions = model.predict(X).squeeze()
    predictions = predictions[1:]  # remove SOS prediction

    output = ''
    confidences = []

    for char, prediction in zip(remove_diacritics(line), predictions):
        output += char
        if char not in a_l_l:
            continue

        max_prob = np.max(prediction)
        predicted_class = r_c_m[np.argmax(prediction)]

        if '<' in predicted_class:
            continue

        if max_prob > confidence_threshold:
            output += predicted_class

        confidences.append(max_prob)

    return output, np.mean(confidences) if confidences else 0.0

def evaluate_predictions(sentences, model):
    total_conf = 0
    results = []

    for sentence in sentences:
        cleaned = clean_text(sentence.strip())
        predicted, conf = predict_with_confidence(cleaned, model)
        print("----")
        
        results.append({
            'original': sentence.strip(),
            'predicted': predicted,
            'confidence': conf
        })
        total_conf += conf

    return results, total_conf / len(results)

# Test examples
test_sentences = [
    "ذهب الولد الي المدرسة",
    "العصفور فوق الشجرة الكبيرة",
    "الكتاب على الطاولة",
    "السماء صافية اليوم",
    "الشمس مشرقة",
    "القمر منير في الليل",
    "البحر واسع وعميق",
    "الجبال شاهقة وعظيمة",
    "النهر يجري بين السهول",
    "الورد جميل ورائحته عطرة",
    'سطعت شمس والجو جميل هذا اليوم , وفراشات قد قال هذا فصل الصيف'
]

# Load trained model
model = load_model(path+"/improved_diacritization.h5")

# Evaluate
print("\n" + "="*50)
print("TESTING IMPROVED DIACRITIZATION MODEL")
print("="*50)

results, avg_conf = evaluate_predictions(test_sentences, model)

for r in results:
    print(f"Original : {r['original']}")
    print(f"Predicted: {r['predicted']}")
    print(f"Conf.    : {r['confidence']:.3f}")
    print("-" * 40)

print(f"\nAverage Confidence: {avg_conf:.3f}")


