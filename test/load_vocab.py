def _load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    for idx, word in enumerate(words):
        vocab[word] = idx
    required_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
    max_idx = max(vocab.values()) if vocab else -1
    for token in required_tokens:
        if token not in vocab:
            max_idx += 1
            vocab[token] = max_idx
    return vocab
