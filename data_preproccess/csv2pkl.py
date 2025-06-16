import pandas as pd
import pickle
import os

def csv_to_pickle(csv_file, pickle_file):
    # Đọc CSV
    print(f"Đang đọc {csv_file}...")
    df = pd.read_csv(csv_file, encoding='utf-8', header=None, names=['ID', 'Sentence'])
    sentences = [str(s).strip() for s in df['Sentence'].tolist() if str(s).strip()]
    
    # Lưu vào pickle
    print(f"Đang lưu vào {pickle_file}...")
    with open(pickle_file, 'wb') as f:
        pickle.dump(sentences, f)
    print(f"Đã lưu {len(sentences)} câu vào {pickle_file}")

# Đường dẫn file
data_dir = 'data/data'
train_source_csv = f'{data_dir}/train/source.csv'
train_target_csv = f'{data_dir}/train/target.csv'
val_source_csv = f'{data_dir}/val/source.csv'
val_target_csv = f'{data_dir}/val/target.csv'
test_source_csv = f'{data_dir}/test/source.csv'
test_target_csv = f'{data_dir}/test/target.csv'

# Tạo thư mục lưu pickle
pickle_dir = 'data/data'
os.makedirs(pickle_dir, exist_ok=True)

# Chuyển đổi
csv_to_pickle(train_source_csv, f'{pickle_dir}/train_source.pkl')
csv_to_pickle(train_target_csv, f'{pickle_dir}/train_target.pkl')
csv_to_pickle(val_source_csv, f'{pickle_dir}/val_source.pkl')
csv_to_pickle(val_target_csv, f'{pickle_dir}/val_target.pkl')
csv_to_pickle(test_source_csv, f'{pickle_dir}/test_source.pkl')
csv_to_pickle(test_target_csv, f'{pickle_dir}/test_target.pkl')