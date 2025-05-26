import torch
import os
import sys
os.chdir(sys.path[0])
from  main2 import Encoder , Decoder , Seq2Seq

checkpoint = torch.load('arabic_pattern_model.pth')

encoder = Encoder(len(checkpoint['input_vocab']), emb_dim=32, hidden_dim=64)
decoder = Decoder(len(checkpoint['target_vocab']), emb_dim=32, hidden_dim=64)
model = Seq2Seq(encoder, decoder, checkpoint['target_vocab'])


model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

input_vocab = checkpoint['input_vocab']
target_vocab = checkpoint['target_vocab']

def infer(model, word, input_vocab, target_vocab, max_len=16):
    model.eval()
    
    # عكس قاموس target_vocab لتحويل الأرقام إلى حروف
    inv_target_vocab = {v: k for k, v in target_vocab.items()}
    
    # تجهيز الكلمة
    word_ids = [input_vocab[c] for c in word]
    word_tensor = torch.tensor(word_ids).unsqueeze(0)  # شكل [1, length]
    
    # تمرير عبر الـ encoder
    with torch.no_grad():
        hidden, cell = model.encoder(word_tensor)
        input_id = torch.tensor([target_vocab['<sos>']])
        output_chars = []
        
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_id, hidden, cell)
            top1 = output.argmax(1).item()
            char = inv_target_vocab.get(top1, '?')

            if char == '<eos>':
                break

            output_chars.append(char)
            input_id = torch.tensor([top1])

    return ''.join(output_chars)


new_word = "يشرب"
predicted_pattern = infer(model, new_word, input_vocab, target_vocab)
print(f"weight : {predicted_pattern}")
