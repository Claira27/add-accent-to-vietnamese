import pandas as pd
import re
import string

# Đường dẫn tới file CSV đầu vào và đầu ra
input_file = 'split_sentences.csv'
output_file = 'cleaned_data.csv'

# Đọc dữ liệu
df = pd.read_csv(input_file)

# Hàm làm sạch câu
def clean_text(text):
    if pd.isna(text):
        return ""
    # Xoá số
    text = re.sub(r'\d+', '', text)
    # Xoá dấu câu (bao gồm cả các dấu đặc biệt kiểu tiếng Việt hay dùng)
    punctuation = string.punctuation + '“”‘’…–—'
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)
    # Chuyển thành chữ thường
    text = text.lower()
    # Loại bỏ khoảng trắng dư thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Làm sạch câu
df['Sentence'] = df['Sentence'].apply(clean_text)

# Loại bỏ câu có ít hơn 5 từ hoặc hơn 100 từ
df = df[df['Sentence'].apply(lambda x: 5 <= len(x.split()) <= 100)]

# Ghi kết quả ra file mới
df.to_csv(output_file, index=False)
print("Đã xử lý và lưu vào", output_file)
