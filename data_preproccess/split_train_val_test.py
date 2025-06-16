import random

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

def split_data(source_path, target_path, seed=42):
    source_lines = load_data(source_path)
    target_lines = load_data(target_path)

    assert len(source_lines) == len(target_lines), "Số dòng source và target không khớp"

    data = list(zip(source_lines, target_lines))
    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    n_train = int(0.8 * total)
    n_val = int(0.1 * total)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    write_file("train_source.csv", [x[0] for x in train])
    write_file("train_target.csv", [x[1] for x in train])
    write_file("val_source.csv", [x[0] for x in val])
    write_file("val_target.csv", [x[1] for x in val])
    write_file("test_source.csv", [x[0] for x in test])
    write_file("test_target.csv", [x[1] for x in test])

    print(f"Tổng: {total} câu")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

split_data('data/source.csv', 'data/target.csv')
