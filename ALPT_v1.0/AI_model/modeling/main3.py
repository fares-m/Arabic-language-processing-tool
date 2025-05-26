import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import time
import datetime
from sklearn.model_selection import train_test_split

# --------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ArabicPatternDataset(Dataset):
    def __init__(self, data_pairs, input_vocab, target_vocab):
        self.data_pairs = data_pairs
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        word, pattern = self.data_pairs[idx]
        word_ids    = [self.input_vocab [c] for c in word ]   
        pattern_ids = [self.target_vocab[c] for c in pattern] ;pattern_ids.append( self.target_vocab["<eos>"])
        return torch.tensor(word_ids).to(device) , torch.tensor(pattern_ids).to(device)



class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [B, dec_hidden_dim]
        # encoder_outputs: [B, src_len, enc_hidden_dim]
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [B, src_len]
        return torch.softmax(attention, dim=1)


# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, enc_hidden_dim, num_layers=num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, src):
        embedded = self.embedding(src)  # [B, src_len, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [B, src_len, enc_hidden_dim*2]
        return outputs, hidden, cell


# --- Decoder ---
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
        # input: [B] (current char)
        input = input.unsqueeze(1)  # [B, 1]
        embedded = self.dropout(self.embedding(input))  # [B, 1, emb_dim]

        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [B, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [B, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [B, 1, enc_hidden_dim*2]

        rnn_input = torch.cat((embedded, context), dim=2)  # [B, 1, emb+ctx]
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        output = output.squeeze(1)       # [B, dec_hidden]
        context = context.squeeze(1)     # [B, ctx]
        embedded = embedded.squeeze(1)   # [B, emb_dim]

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
def build_vocab(samples):
    chars = set()
    for w, p in samples:
        chars.update(w)
        chars.update(p)
    char_list = sorted(list(chars)) + ['<pad>', '<sos>', '<eos>']
    vocab = {c: i for i, c in enumerate(char_list)}
    return vocab

def pad_batch(batch, input_vocab, target_vocab):
    from torch.nn.utils.rnn import pad_sequence
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=input_vocab['<pad>'])
    targets = pad_sequence(targets, batch_first=True, padding_value=target_vocab['<pad>'])
    return inputs.to(device) , targets.to(device)

def evaluate_accuracy(model, val_loader, target_vocab):
    model.eval()
    correct = 0
    total = 0
    eos_token = target_vocab['<eos>']
    with torch.no_grad():
        for src, trg in val_loader:
            outputs = model(src, trg, teacher_forcing_ratio=0.0)
            pred_tokens = outputs.argmax(-1)

            for pred_seq, true_seq in zip(pred_tokens, trg):
                pred_seq = pred_seq.tolist()
                true_seq = true_seq.tolist()
                if eos_token in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(eos_token)]
                if eos_token in true_seq:
                    true_seq = true_seq[:true_seq.index(eos_token)]

                if pred_seq == true_seq:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


# load the word-weigth dict 
with open(r"D:\py\final_project\AI_model\data_collection_and_preprocessing\data_collection\new_space\final_dict_triple.pickle" , 'rb') as file :
    words_dict:dict = pickle.load(file)

if __name__ == "__main__":
    input_words  = list(words_dict.keys())
    output_words = list(words_dict.values())
    print("data loaded.")

    samples = list(zip(input_words , output_words))
    train_samples, val_samples = train_test_split(samples, test_size=0.1, random_state=42)

    input_vocab = build_vocab(samples) 
    target_vocab = build_vocab(samples)
    
    train_dataset = ArabicPatternDataset(train_samples, input_vocab, target_vocab)
    val_dataset   = ArabicPatternDataset(val_samples, input_vocab, target_vocab)

    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=lambda b: pad_batch(b, input_vocab, target_vocab))
    val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=lambda b: pad_batch(b, input_vocab, target_vocab))

    print("data prrocessed.")



if __name__ == "__main__":
    emb_dim = 16
    enc_hidden_dim = 32
    dec_hidden_dim = 32
    attention = Attention(enc_hidden_dim * 2, dec_hidden_dim)  # encoder is bidirectional
    encoder = Encoder(len(input_vocab), emb_dim, enc_hidden_dim)
    decoder = Decoder(len(target_vocab), emb_dim, enc_hidden_dim, dec_hidden_dim, attention)
    model = Seq2Seq(encoder, decoder, sos_token=target_vocab['<sos>'], eos_token=target_vocab['<eos>'], device=device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab['<pad>'])


if __name__ == "__main__":
    print(f"start training {datetime.datetime.now()}")
    t1 = time.time()
    for epoch in range(1):
        try :
            model.train()
            total_loss = []
            batch_num = len(train_loader)
            counter = 0
            for src, trg in train_loader:

                optimizer.zero_grad()
                output = model(src, trg)
                output = output.view(-1, output.shape[-1])
                trg = trg.view(-1)
                loss = criterion(output, trg)
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
                
                counter += 1
                
                if counter % float(batch_num/10).__ceil__()==0 :
                    print(f"[+]Time:{datetime.datetime.now()}|Epoch:{epoch+1}|Loss:{sum(total_loss)/len(total_loss):.4f}|Time:{time.time()-t1:.3f}s|complete:%{counter/batch_num*100:.2f}|")
            val_acc = evaluate_accuracy(model, val_loader, target_vocab)
            print(f"[+]Time:{datetime.datetime.now()}|Epoch:{epoch+1}|Loss:{sum(total_loss)/len(total_loss):.4f}|Time:{time.time()-t1:.3f}s|  [Val Accuracy:{val_acc*100:.2f}%]|\t epoch {epoch+1} complited.")
            

        except KeyboardInterrupt :
            print("[-] KeyboardInterrupt Except.")
            print("saving modle ....")
            print("exit.")
            break
        

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_vocab': input_vocab,
        'target_vocab': target_vocab,
    }, 'arabic_pattern_model.pth')

