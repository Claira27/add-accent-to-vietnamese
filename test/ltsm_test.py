import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from load_vocab import _load_vocab
from denormalize import denormalize_tone_line
import os
import torch.serialization

# Định nghĩa các lớp mô hình LSTM
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(EncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=in_vocab['<pad>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        stdv = 1. / (self.v.size(0) ** 0.5)
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        return F.softmax(attention, dim=1)

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=out_vocab['<pad>'])
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)

    def forward(self, tgt, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(tgt))
        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell, attn_weights

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.8):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = tgt[:, 0]
        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(input.unsqueeze(1), hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
        return outputs

# Tải từ vựng
in_vocab = _load_vocab("demoapp/utils/input_vocab.txt")
out_vocab = _load_vocab("demoapp/utils/output_vocab.txt")

# Hàm suy luận với beam search
def beam_search(model, src, in_vocab, out_vocab, max_len, device, beam_size=3):
    model.eval()
    src = src.to(device)
    start_token = out_vocab['<sos>']
    end_token = out_vocab['<eos>']
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)
    sequences = [[torch.tensor([start_token], dtype=torch.long, device=device), 0.0, hidden, cell]]
    completed_sequences = []
    for _ in range(max_len):
        all_candidates = []
        for seq, score, hidden, cell in sequences:
            if seq[-1].item() == end_token:
                completed_sequences.append([seq, score])
                continue
            with torch.no_grad():
                output, hidden, cell, _ = model.decoder(seq[-1].unsqueeze(0).unsqueeze(1), hidden, cell, encoder_outputs)
            probs = F.softmax(output.squeeze(1), dim=1)
            top_probs, top_indices = probs.topk(beam_size, dim=1)
            for i in range(beam_size):
                token = top_indices[0, i].unsqueeze(0)
                token_prob = top_probs[0, i].item()
                new_seq = torch.cat((seq, token), dim=0)
                new_score = score + torch.log(torch.tensor(token_prob))
                all_candidates.append([new_seq, new_score, hidden, cell])
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = all_candidates[:beam_size]
        if len(sequences) == 0:
            break
    if completed_sequences:
        best_seq = max(completed_sequences, key=lambda x: x[1])[0]
    else:
        best_seq = sequences[0][0]
    idx2word = {idx: word for word, idx in out_vocab.items()}
    translated = [idx2word.get(token.item(), '<unk>') for token in best_seq[1:] if token.item() != end_token]
    return translated

# Hàm dịch câu
def translate(model, sentence, max_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokens = sentence.strip().split()
    token_ids = [in_vocab.get(token, in_vocab['<unk>']) for token in tokens]
    token_ids = token_ids[:max_len] + [in_vocab['<pad>']] * (max_len - len(token_ids))
    src = torch.tensor([token_ids], dtype=torch.long).to(device)
    pred_tokens = beam_search(model, src, in_vocab, out_vocab, max_len, device, beam_size=3)
    normalized_sentence = ' '.join(pred_tokens)
    denormalized_sentence = denormalize_tone_line(normalized_sentence)
    return denormalized_sentence

# Khởi tạo và tải mô hình
embedding_dim = 256
hidden_dim = 512
num_layers = 1
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Allowlist the Seq2SeqLSTM class
torch.serialization.add_safe_globals([Seq2SeqLSTM])

encoder = EncoderLSTM(len(in_vocab), embedding_dim, hidden_dim, num_layers, dropout)
decoder = DecoderLSTM(len(out_vocab), embedding_dim, hidden_dim, num_layers, dropout)
model = Seq2SeqLSTM(encoder, decoder, device).to(device)

# Tải trọng số mô hình
model_path = "demoapp/models/lstm_model.pt"
if os.path.exists(model_path):

    model = torch.load(model_path, map_location=device, weights_only=False)
    # model = {k.replace('module.', ''): v for k, v in model.items()}  # Xử lý nếu mô hình được lưu với 'module.' prefix
    # model.load_state_dict(model)
    model.eval()
else:
    raise FileNotFoundError(f"Model file {model_path} not found.")

input_sentence = "tieng viet duoc chinh thuc ghi nhan trong hien phap cua nuoc cong hoa xa hoi chu nghia viet nam"
output_sentence = translate(model, input_sentence)
print(f"Input: {input_sentence}")   
print(f"Output: {output_sentence}")