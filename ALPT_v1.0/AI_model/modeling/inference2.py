import torch
import torch.nn as nn
import pickle

import os
import sys
os.chdir(sys.path[0])

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, enc_hidden_dim, num_layers=num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, attention, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(enc_hidden_dim*2 + emb_dim, dec_hidden_dim, num_layers=4, batch_first=True)
        self.fc_out = nn.Linear(enc_hidden_dim*2 + dec_hidden_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_token, eos_token, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)

        input = torch.tensor([self.sos_token] * batch_size).to(self.device)

        for t in range(trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs
    
    def predict(self, src):
        """
        تستخدم للتنبؤ بالوزن الصرفي للكلمة المدخلة
        """
        self.eval()  # وضع التقييم
        
        with torch.no_grad():
            batch_size = src.size(0)
            
            # الحصول على مخرجات المشفر
            encoder_outputs, hidden, cell = self.encoder(src)
            
            # تهيئة المخرجات
            predicted_pattern = []
            
            # البدء برمز بداية التسلسل
            input_token = torch.tensor([self.sos_token] * batch_size).to(self.device)
            
            # الحد الأقصى لطول التنبؤ - نستخدم قيمة مناسبة لأوزان الكلمات العربية
            max_len = 15
            
            for i in range(max_len):
                # الحصول على التنبؤ من فك التشفير
                output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
                
                # الحصول على الرمز الأكثر احتمالاً
                top_token = output.argmax(1)
                
                # إضافة الرمز إلى قائمة التنبؤات
                predicted_token = top_token.item()
                predicted_pattern.append(predicted_token)
                
                # التحقق مما إذا كان الرمز هو رمز نهاية التسلسل
                if predicted_token == self.eos_token:
                    break
                
                # استخدام التنبؤ الحالي كمدخل للخطوة التالية
                input_token = top_token
            
            return predicted_pattern


def load_model(model_path, device):
    """
    تحميل النموذج المحفوظ ومعاجمه
    """
    # تحميل البيانات المحفوظة
    checkpoint = torch.load(model_path, map_location=device)
    
    # استخراج المعاجم
    input_vocab = checkpoint['input_vocab']
    target_vocab = checkpoint['target_vocab']
    
    # إنشاء معاجم عكسية
    input_idx_to_char = {idx: char for char, idx in input_vocab.items()}
    target_idx_to_char = {idx: char for char, idx in target_vocab.items()}
    
    
    
    # إعادة إنشاء النموذج بنفس المعلمات المستخدمة في التدريب
    emb_dim = 64
    enc_hidden_dim = 256
    dec_hidden_dim = 256
    
    attention = Attention(enc_hidden_dim * 2, dec_hidden_dim)
    encoder = Encoder(len(input_vocab), emb_dim, enc_hidden_dim)
    decoder = Decoder(len(target_vocab), emb_dim, enc_hidden_dim, dec_hidden_dim, attention)
    
    model = Seq2Seq(encoder, decoder, 
                   sos_token=target_vocab['<sos>'], 
                   eos_token=target_vocab['<eos>'], 
                   device=device)
    
    # تحميل حالة النموذج
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # وضع التقييم للاستدلال
    
    return model, input_vocab, target_vocab, input_idx_to_char, target_idx_to_char


def convert_word_to_pattern(model, word, input_vocab, target_vocab, target_idx_to_char, device):
    """
    تحويل كلمة عربية إلى وزنها الصرفي
    """
    
    # التحقق من وجود الأحرف في المعجم
    unknown_chars = [c for c in word if c not in input_vocab]
    if unknown_chars:
        print(f"unknown_chars : {unknown_chars}")
    
    # استخدام رمز <pad> للأحرف غير المعروفة
    word_indices = [input_vocab.get(c, input_vocab['<pad>']) for c in word]
    word_tensor = torch.tensor([word_indices], device=device)
    
    print(f"word : {word_indices}")
    
    # الحصول على التنبؤ من النموذج
    predicted_indices = model.predict(word_tensor)
    
    print(f"weight : {predicted_indices}")
    
    # تحويل المؤشرات إلى أحرف باستخدام معجم المخرجات العكسي
    pattern_chars = []
    for idx in predicted_indices:
        if idx == target_vocab['<eos>']:
            break  # التوقف عند رمز النهاية
        if idx != target_vocab['<sos>']:  # تجاهل رمز البداية
            pattern_chars.append(target_idx_to_char[idx])
    
    # تجميع الأحرف لتكوين الوزن
    pattern = ''.join(pattern_chars)
    
    return pattern

def fix_weight(word , weight):
    word_len = len(word)
    weight_len = len(weight)

    if word_len == weight_len and ('\x00' not in weight) and  ('\x01' not in weight):
        return weight
    new_weight = []
    counter = 0
    for index , i in enumerate(weight):
        if i == '\x00' :word_len += 1
        
        if i not in ['\x01' , '\x00'] :
            new_weight.append(i)
            counter +=1
        else :new_weight.append(i)
        if counter == word_len :
            return new_weight
    
def main():
    # تحديد الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # تحميل النموذج
    model_path = 'arabic_pattern_model_vv0.pth'
    model, input_vocab, target_vocab, input_idx_to_char, target_idx_to_char = load_model(model_path, device)
    
    # قائمة الكلمات للاختبار
    test_words = ["الأصدقاء"]
    
    
    for word in test_words:
        try:
            pattern = convert_word_to_pattern(model, word, input_vocab, target_vocab, target_idx_to_char, device)
            print(f"word: {word} -> weigth : {pattern}")
        except Exception as e:
            print(f"Error :  {e}")
        


# تنفيذ البرنامج
if __name__ == "__main__":
    main()