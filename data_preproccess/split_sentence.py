import re
import pandas as pd

# Hàm tách câu
def split_sentences(text):
    pattern = r'(?<=[.;:?!])\s+(?=[^\d])|(?<=\.\.\.)\s+'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]

# Đọc file TXT
input_file = "raw_data/train_tieng_viet.txt"
result = []
try:
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                print(f"Dòng không hợp lệ: {line.strip()}")
                continue
            id, text = parts
            sentences = split_sentences(text)
            for i, sentence in enumerate(sentences, 1):
                result.append({"ID": f"{id}_{i:03d}", "Sentence": sentence})
except FileNotFoundError:
    print(f"File {input_file} không tồn tại. Vui lòng kiểm tra đường dẫn.")
    exit()

# Chuyển thành DataFrame và lưu
result_df = pd.DataFrame(result)
output_file = "split_sentences.csv"
result_df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Kết quả đã được lưu vào {output_file}")

# In kết quả
for _, row in result_df.iterrows():
    print(f"{row['ID']}: {row['Sentence']}")