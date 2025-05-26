import time
import numpy as np
import pickle as pkl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, TimeDistributed, BatchNormalization, Bidirectional
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import re
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle

#__________________load helper files_________________

WITH_EXTRA_TRAIN = False

with open('ARABIC_LETTERS_LIST.pickle', 'rb') as file:
    a_l_l = pkl.load(file)

with open('DIACRITICS_LIST.pickle', 'rb') as file:
    d_l = pkl.load(file)

if not WITH_EXTRA_TRAIN:
    with open('RNN_SMALL_CHARACTERS_MAPPING.pickle', 'rb') as file:
        char_m = pkl.load(file)
else:
    with open('RNN_BIG_CHARACTERS_MAPPING.pickle', 'rb') as file:
        char_m = pkl.load(file)

with open('RNN_CLASSES_MAPPING.pickle', 'rb') as file:
    class_m = pkl.load(file)

with open('RNN_REV_CLASSES_MAPPING.pickle', 'rb') as file:
    r_c_m = pkl.load(file)

#________________________load data _________________________

train_data_raw = None
with open('train_extra_data.txt', 'r', encoding='utf-8') as file:
    train_data_raw = file.readlines()

val_data_raw = None
with open('val_add.txt', 'r', encoding='utf-8') as file:
    val_data_raw = file.readlines()

#_____________________clean data _________________

def remove_diacritics(data):
    return "".join([char for char in data if char not in d_l])

def clean_text(text):
    text = re.sub(r"[-()\"#/@;:_<>{}`﴾﴿ے+=~&|.!?,a-zA-Z0-9٠-٩…''•]", "", text)
    return text

def split_data(data_raw):
    data_new = list()
    
    for line in data_raw:
        line = line.replace('.', '.\n')
        line = line.replace(',', ',\n')
        line = line.replace('،', '،\n')
        line = line.replace(':', ':\n')
        line = line.replace(';', ';\n')
        line = line.replace('؛', '؛\n')
        line = line.replace('(', '\n(')
        line = line.replace(')', ')\n')
        line = line.replace('[', '\n[')
        line = line.replace(']', ']\n')
        line = line.replace('{', '\n{')
        line = line.replace('}', '}\n')
        line = line.replace('«', '\n«')
        line = line.replace('»', '»\n')

        for sub_line in line.split('\n'):
            if len(remove_diacritics(sub_line).strip()) == 0:
                continue
            if len(remove_diacritics(sub_line).strip()) > 0 and len(remove_diacritics(sub_line).strip()) <= 500:
                data_new.append(sub_line.strip())
            else:
                sub_line = sub_line.split()
                tmp_line = ''
                for word in sub_line:
                    if len(remove_diacritics(tmp_line).strip()) + len(remove_diacritics(word).strip()) + 1 > 500:
                        if len(remove_diacritics(tmp_line).strip()) > 0:
                            data_new.append(tmp_line.strip())
                        tmp_line = word
                    else:
                        if tmp_line == '':
                            tmp_line = word
                        else:
                            tmp_line += ' '
                            tmp_line += word
                if len(remove_diacritics(tmp_line).strip()) > 0:
                    data_new.append(tmp_line.strip())

    return data_new

def map_data(data_raw):
    X = list()
    Y = list()

    for line in data_raw:
        x = [char_m['<SOS>']]
        y = [class_m['<SOS>']]

        for idx, char in enumerate(line):
            if char in d_l:
                continue
            x.append(char_m[char])

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
                y.append(class_m[char_diac])

        assert(len(x) == len(y))

        x.append(char_m['<EOS>'])
        y.append(class_m['<EOS>'])

        y = to_categorical(y, len(class_m))

        X.append(x)
        Y.append(y)

    return X, Y

#_____________________________________

train_split = split_data(train_data_raw)
val_split = split_data(val_data_raw)

for i in range(len(train_split)):
    train_split[i] = clean_text(train_split[i])

for i in range(len(val_split)):
    val_split[i] = clean_text(val_split[i])

train_split = shuffle(train_split, random_state=42)

#____________________________ Improved modeling ____________________________

model = Sequential()

model.add(Embedding(input_dim=len(char_m), output_dim=64, mask_zero=True))
model.add(Dropout(0.4))

model.add(Bidirectional(LSTM(128, return_sequences=True, 
                             kernel_regularizer=l2(0.001))))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(64, return_sequences=True,
                             kernel_regularizer=l2(0.001))))
model.add(Dropout(0.4))

model.add(TimeDistributed(Dense(128, activation='relu', 
                               kernel_regularizer=l2(0.002))))
model.add(Dropout(0.3))

model.add(TimeDistributed(Dense(64, activation='relu',
                               kernel_regularizer=l2(0.002))))
model.add(Dropout(0.2))

model.add(TimeDistributed(Dense(len(class_m), activation='softmax')))

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#_________________________________

class ImprovedDataGenerator(Sequence):
    def __init__(self, lines, batch_size, shuffle=True):
        self.lines = lines
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.lines) / float(self.batch_size)))

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        lines = [self.lines[i] for i in indices]
        X_batch, Y_batch = map_data(lines)

        X_max_seq_len = np.max([len(x) for x in X_batch])
        Y_max_seq_len = np.max([len(y) for y in Y_batch])

        assert(X_max_seq_len == Y_max_seq_len)

        X = list()
        for x in X_batch:
            x = list(x)
            x.extend([char_m['<PAD>']] * (X_max_seq_len - len(x)))
            X.append(np.asarray(x))

        Y_tmp = list()
        for y in Y_batch:
            y_new = list(y)
            y_new.extend(to_categorical([class_m['<PAD>']] * (Y_max_seq_len - len(y)), len(class_m)))
            Y_tmp.append(np.asarray(y_new))
        Y_batch = Y_tmp

        Y_batch = np.array(Y_batch)

        return (np.array(X), Y_batch)
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.lines))
        if self.shuffle:
            np.random.shuffle(self.indices)

#________________________

training_generator = ImprovedDataGenerator(train_split, 64, shuffle=True)
val_generator = ImprovedDataGenerator(val_split, 64, shuffle=False)

# Define callbacks to prevent overfitting
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_diacritization_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

start_time = time.time()

hist_1 = model.fit(
    x=training_generator,
    validation_data=val_generator,
    epochs=15, 
    callbacks=callbacks,
    verbose=1
)

end_time = time.time()

print('--- %f seconds ---' % round(end_time - start_time, 2))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(hist_1.history['loss'])
plt.plot(hist_1.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(hist_1.history['accuracy'])
plt.plot(hist_1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()

# Save the final model
model.save("improved_diacritization.h5")

def predict_with_confidence(line, model, confidence_threshold=0.5):
    
    X, _ = map_data([line])
    X = np.asarray(X)
    predictions = model.predict(X).squeeze()
    predictions = predictions[1:]  

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
   
    total_confidence = 0
    results = []
    
    for sentence in sentences:
        predicted, confidence = predict_with_confidence(sentence, model)
        results.append({
            'original': sentence,
            'predicted': predicted,
            'confidence': confidence
        })
        total_confidence += confidence
    
    avg_confidence = total_confidence / len(sentences)
    return results, avg_confidence

# Test samples
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
    "الفاكهة لذيذة ومفيدة",
    "الخبز طعام أساسي",
    "الماء ضروري للحياة",
    "الهواء نقي في الصباح",
    "النار تعطي الدفء"
]

print("\n" + "="*50)
print("IMPROVED MODEL PREDICTIONS WITH CONFIDENCE")
print("="*50)

results, avg_confidence = evaluate_predictions(test_sentences, model)

for result in results:
    print(f"Original:   {result['original']}")
    print(f"Predicted:  {result['predicted']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("-" * 40)

print(f"\nAverage Confidence: {avg_confidence:.3f}")

model.summary()

