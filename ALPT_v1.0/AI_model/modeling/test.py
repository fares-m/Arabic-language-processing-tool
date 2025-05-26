import torch
import torch.nn as nn
import pickle

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        embedded_src = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)

        embedded_tgt = self.embedding(tgt)
        output, _ = self.decoder(embedded_tgt, (hidden, cell))
        return self.fc(output)

# تحميل المعاجم
with open(r"D:\py\final_project\AI_model\modeling\dicts.pickle", "rb") as f:
    char2idx, idx2char, max_len = pickle.load(f)

vocab_size = len(char2idx)

# تحميل النموذج كاملاً
model = torch.load(r"D:\py\final_project\AI_model\modeling\seq2seq_model2.model", map_location='cpu')
model.eval()

# دالة لتحويل الكلمة إلى Tensor
def word_to_tensor(word):
    indices = [char2idx.get(ch, 0) for ch in word]
    indices += [char2idx["<PAD>"]] * (max_len - len(indices))
    return torch.tensor(indices).unsqueeze(0)  # شكل: [1, max_len]

# دالة لتحويل التنسور الناتج إلى كلمة
def tensor_to_word(tensor):
    indices = tensor.argmax(dim=-1).squeeze().tolist()
    return "".join(idx2char.get(idx, "") for idx in indices if idx != char2idx["<PAD>"])

# واجهة اختبار بسيطة
while True:
    word = input("أدخل كلمة لاختبار النموذج (أو 'خروج'): ").strip()
    if word.lower() == "خروج":
        break
    if len(word) > max_len:
        print(f"⚠️ الكلمة أطول من الحد الأقصى ({max_len})")
        continue

    src_tensor = word_to_tensor(word)
    tgt_tensor = torch.zeros_like(src_tensor)  # وهمي فقط

    with torch.no_grad():
        output = model(src_tensor, tgt_tensor)

    predicted_word = tensor_to_word(output)
    print(f"🔁 الكلمة المتوقعة: {predicted_word}")
