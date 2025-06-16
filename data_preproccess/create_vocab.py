import re

def remove_tone_line(utf8_str):
    intab_l = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
    intab_u = "ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    intab = list(intab_l + intab_u)

    outtab_l = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"
    outtab_u = "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"
    outtab = outtab_l + outtab_u

    replaces_dict = dict(zip(intab, outtab))

    def replace_char(ch):
        return replaces_dict.get(ch, ch)

    return ''.join(replace_char(ch) for ch in utf8_str)


def normalize_tone_line(utf8_str):
    # Áp dụng chuẩn normalize dấu dạng mã 
    intab_l = "áàảãạâấầẩẫậăắằẳẵặđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    intab_u = "ÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
    intab = list(intab_l + intab_u)

    outtab_l = [
        "a1", "a2", "a3", "a4", "a5",
        "a6", "a61", "a62", "a63", "a64", "a65",
        "a8", "a81", "a82", "a83", "a84", "a85",
        "d9",
        "e1", "e2", "e3", "e4", "e5",
        "e6", "e61", "e62", "e63", "e64", "e65",
        "i1", "i2", "i3", "i4", "i5",
        "o1", "o2", "o3", "o4", "o5",
        "o6", "a61", "o62", "o63", "o64", "o65",
        "o7", "o71", "o72", "o73", "o74", "o75",
        "u1", "u2", "u3", "u4", "u5",
        "u7", "u71", "u72", "u73", "u74", "u75",
        "y1", "y2", "y3", "y4", "y5",
    ]

    outtab_u = [
        "A1", "A2", "A3", "A4", "A5",
        "A6", "A61", "A62", "A63", "A64", "A65",
        "A8", "A81", "A82", "A83", "A84", "A85",
        "D9",
        "E1", "E2", "E3", "E4", "E5",
        "E6", "E61", "E62", "E63", "E64", "E65",
        "I1", "I2", "I3", "I4", "I5",
        "O1", "O2", "O3", "O4", "O5",
        "O6", "O61", "O62", "O63", "O64", "O65",
        "O7", "O71", "O72", "O73", "O74", "O75",
        "U1", "U2", "U3", "U4", "U5",
        "U7", "U71", "U72", "U73", "U74", "U75",
        "Y1", "Y2", "Y3", "Y4", "Y5",
    ]

    replaces_dict = dict(zip(intab, outtab_l + outtab_u))

    def replace_char(ch):
        return replaces_dict.get(ch, ch)

    return ''.join(replace_char(ch) for ch in utf8_str)


def main():
    input_file = 'raw_data/Viet22K.txt'  

    words_no_diacritic = set()
    words_tone_code = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            # tách từ đơn trong câu (các từ phân tách bởi space)
            tokens = re.findall(r'\b\w+\b', line, flags=re.UNICODE)
            for token in tokens:
                # tạo từ không dấu
                no_diacritic = remove_tone_line(token)
                words_no_diacritic.add(no_diacritic)

                # tạo từ có dấu dạng mã
                tone_code = normalize_tone_line(token)
                words_tone_code.add(tone_code)

    # lưu ra file
    with open('data/source_vocab.txt', 'w', encoding='utf-8') as f_out1:
        for w in sorted(words_no_diacritic):
            f_out1.write(w + '\n')

    with open('data/target_vocab.txt', 'w', encoding='utf-8') as f_out2:
        for w in sorted(words_tone_code):
            f_out2.write(w + '\n')

    print(f"Số từ không dấu: {len(words_no_diacritic)}")
    print(f"Số từ dạng mã: {len(words_tone_code)}")

if __name__ == '__main__':
    main()
