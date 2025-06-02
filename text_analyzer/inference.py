import torch
import torch.nn as nn


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
            max_len = 18
            
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # استخراج المعاجم
    input_vocab = checkpoint['input_vocab']
    target_vocab = checkpoint['target_vocab']
    
    # إنشاء معاجم عكسية
    input_idx_to_char = {idx: char for char, idx in input_vocab.items()}
    target_idx_to_char = {idx: char for char, idx in target_vocab.items()}
    
    
    
    # إعادة إنشاء النموذج بنفس المعلمات المستخدمة في التدريب
    emb_dim = 64
    enc_hidden_dim = 128
    dec_hidden_dim = 128
    
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
    if unknown_chars:print(f"unknown_chars : {unknown_chars}")
    
    # استخدام رمز <pad> للأحرف غير المعروفة
    word_indices = [input_vocab.get(c, input_vocab['<pad>']) for c in word]
    word_tensor = torch.tensor([word_indices], device=device)
    
    
    
    # الحصول على التنبؤ من النموذج
    predicted_indices = model.predict(word_tensor)
    
    
    
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


def fix_weight(word:str , weight:str):
    word_len = len(word)

    # if no 0x00 0x01 in weight mean that word len must equal weight len
    if ("\x00" not in weight) and  ("\x01" not in weight): 
        fixed_weight = weight[:word_len]
        # simple condition (if the three origin letter not exist at least once return -1 )
        if (fixed_weight.count('ف') + fixed_weight.count('ع') + fixed_weight.count('ل') ) <3:return -1 
        return fixed_weight
    
    # otherwise (there is a 0x00 OR 0x01 in weight)
    if ("\x00" in weight):word_len +=2 # weight will be longer in 2 char if there is a deleted  letter | ex : يبع : يبي0ع
    if ("\x01" in weight):word_len +=1 # weight will be longer in 1 char if there is a replaced letter | ex : صيام : فو1ال
    fixed_weight = weight[:word_len]
    # spacial char : [ 0x00 , 0x01 ]
    # simple condition (if the three origin letters OR spacial char letter did not occur 3 times at least , return -1 )
    if (fixed_weight.count('ف') + fixed_weight.count('ع') + fixed_weight.count('ل') +
         fixed_weight.count("\x00") + fixed_weight.count("\x01")) < 3:return -1

    """these conditions are a simple condition, will not fix corrupted weights allways"""
    return fixed_weight
    

def extract_weight(path , word:str):
    # تحديد الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # تحميل النموذج
    model_path = path+'/arabic_pattern_model_v3.pth'
    model, input_vocab, target_vocab, input_idx_to_char, target_idx_to_char = load_model(model_path, device )
        
    try:
        pattern = convert_word_to_pattern(model, word, input_vocab, target_vocab, target_idx_to_char, device)
        return pattern
    except Exception as e:
        return None



