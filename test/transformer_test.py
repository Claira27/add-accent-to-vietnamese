import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from load_vocab import _load_vocab
from denormalize import denormalize_tone_line
import os
import string
import torch.serialization

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#Tải từ vựng
in_vocab = _load_vocab("demoapp/utils/input_vocab.txt")
out_vocab = _load_vocab("demoapp/utils/output_vocab.txt")
# Định nghĩa các lớp mô hình Transformer
DIM_MODEL = 512
N_HEADS = 8
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.1
MAX_LEN = 50
NUM_EPOCHS = 6

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None, padding_mask=None):
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)

        query = self.split_heads(self.query_linear(query), batch_size)  # [batch_size, n_heads, query_len, depth]
        key = self.split_heads(self.key_linear(key), batch_size)        # [batch_size, n_heads, key_len, depth]
        value = self.split_heads(self.value_linear(value), batch_size)  # [batch_size, n_heads, key_len, depth]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)  # [batch_size, n_heads, query_len, key_len]

        # Áp dụng padding mask (ngăn attention đến các vị trí pad)
        if padding_mask is not None:
            # padding_mask: [batch_size, key_len]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, key_len]
            attention_scores = attention_scores.masked_fill(padding_mask == 1, float('-inf'))

        # Áp dụng causal mask hoặc source mask (nếu có)
        if mask is not None:
            # mask: [key_len, key_len] hoặc [query_len, key_len]
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, query_len/key_len, key_len]
                mask = mask.repeat(batch_size, self.n_heads, 1, 1)  # [batch_size, n_heads, query_len/key_len, key_len]
            elif len(mask.shape) == 3:
                mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # [batch_size, n_heads, query_len/key_len, key_len]
            # Đảm bảo kích thước của mask phù hợp với attention_scores
            if mask.size(2) != query_len or mask.size(3) != key_len:
                mask = mask[:, :, :query_len, :key_len]
            attention_scores = attention_scores + mask

        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, value)  # [batch_size, n_heads, query_len, depth]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.d_model)
        output = self.out_linear(context)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, padding_mask=None):
        attn_output = self.attention(x, x, x, mask=mask, padding_mask=padding_mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        # Self-attention với target (causal mask)
        self_attn_output = self.self_attention(x, x, x, mask=tgt_mask, padding_mask=tgt_padding_mask)
        x = self.layer_norm1(x + self.dropout(self_attn_output))

        # Cross-attention với encoder output
        cross_attn_output = self.cross_attention(x, enc_output, enc_output, mask=src_mask, padding_mask=src_padding_mask)
        x = self.layer_norm2(x + self.dropout(cross_attn_output))

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + self.dropout(ffn_output))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=5000, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_len, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        return pe

    def forward(self, x, mask=None, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask=mask, padding_mask=padding_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_len, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        return pe

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        logits = self.output_linear(x)
        return logits

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        enc_output = self.encoder(src, mask=src_mask, padding_mask=src_padding_mask)
        logits = self.decoder(tgt, enc_output, src_mask=src_mask, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        return logits

# Generate target mask for causal attention
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Greedy decoding for inference
def greedy_decode(model, src, src_mask, src_padding_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    src_padding_mask = src_padding_mask.to(DEVICE)
    
    memory = model.encoder(src, mask=src_mask, padding_mask=src_padding_mask)
    
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    
    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(DEVICE)
        tgt_padding_mask = (ys == out_vocab['<pad>']).to(DEVICE)
        out = model.decoder(ys, memory, src_mask=src_mask, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        prob = out[:, -1, :]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=DEVICE)], dim=1)
        if next_word == out_vocab['<eos>']:
            break
    return ys

# Vocabulary class for token lookup
class SimpleVocab:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {id: token for token, id in vocab.items()}
    def lookup_tokens(self, ids):
        return [self.inv_vocab.get(i, "<unk>") for i in ids]

out_vocab_transform = SimpleVocab(out_vocab)
in_vocab_transform = SimpleVocab(in_vocab)

# Translation function
def translate(model, src_sentence, max_len=MAX_LEN):
    model.eval()
    src_tensor = simple_text_transform(src_sentence).to(DEVICE)
    seq_len = src_tensor.size(1)
    src_mask = torch.zeros(seq_len, seq_len, device=DEVICE).type(torch.float)
    src_padding_mask = (src_tensor == in_vocab['<pad>']).to(DEVICE)
    
    ys = greedy_decode(model, src_tensor, src_mask, src_padding_mask, max_len=seq_len + 5, start_symbol=out_vocab['<sos>'])
    tgt_tokens = ys.squeeze(0).cpu().numpy().tolist()
    
    tokens = out_vocab_transform.lookup_tokens(tgt_tokens)
    num_words = len(src_sentence.split())
    translation = " ".join(tokens).replace("<sos>", "").replace("<eos>", "").strip()
    if len(translation.split()) > num_words:
        translation = " ".join(translation.split()[:num_words])
    translation = denormalize_tone_line(translation)
    return translation

def simple_text_transform(sentence: str):
    sentence = sentence.strip().lower()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    tokens = sentence.split()
    token_ids = [in_vocab.get(token, in_vocab['<unk>']) for token in tokens]
    token_ids = token_ids[:MAX_LEN] + [in_vocab['<pad>']] * (MAX_LEN - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

# Allowlist the Transformer class
torch.serialization.add_safe_globals([Transformer])

model = Transformer(
    src_vocab_size=len(in_vocab),
    tgt_vocab_size=len(out_vocab),
    d_model=DIM_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_len=50,
    dropout=DROPOUT
).to(DEVICE)

# Tải trọng số mô hình
model_path = "demoapp/models/transformer_model.pt"
if os.path.exists(model_path):
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
else:
    raise FileNotFoundError(f"Model file {model_path} not found.")

input_sentence = "tieng viet duoc chinh thuc ghi nhan trong hien phap cua nuoc cong hoa xa hoi chu nghia viet nam"
output_sentence = translate(model, input_sentence)
print(f"Input: {input_sentence}")
print(f"Output: {output_sentence}")