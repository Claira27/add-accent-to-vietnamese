import csv
import unicodedata
import re

def normalize_tone_line(utf8_str):
    intab_l = u"áàảãạâấầẩẫậăắằẳẵặđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    intab_u = u"ÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
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
        "o6", "o61", "o62", "o63", "o64", "o65",
        "o7", "o71", "o72", "o73", "o74", "o75",
        "u1", "u2", "u3", "u4", "u5",
        "u7", "u71", "u72", "u73", "u74", "u75",
        "y1", "y2", "y3", "y4", "y5",
    ]

    outtab_u = [s.upper() for s in outtab_l]
    outtab = outtab_l + outtab_u

    replaces_dict = dict(zip(intab, outtab))
    r = re.compile("|".join(re.escape(ch) for ch in intab))

    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)

def remove_tone_line(utf8_str):
    return unicodedata.normalize('NFD', utf8_str).encode('ascii', 'ignore').decode('utf-8')

def process_csv(input_path, output_normalized, output_no_tone):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_normalized, 'w', encoding='utf-8', newline='') as f_norm, \
         open(output_no_tone, 'w', encoding='utf-8', newline='') as f_nt:

        reader = csv.DictReader(f_in)
        norm_writer = csv.writer(f_norm)
        no_tone_writer = csv.writer(f_nt)

        norm_writer.writerow(['ID', 'Sentence'])
        no_tone_writer.writerow(['ID', 'Sentence'])

        for row in reader:
            sid = row['ID']
            sentence = row['Sentence']
            norm_sentence = normalize_tone_line(sentence)
            no_tone_sentence = remove_tone_line(sentence)
            norm_writer.writerow([sid, norm_sentence])
            no_tone_writer.writerow([sid, no_tone_sentence])

    print("Done: created", output_normalized, "and", output_no_tone)

if __name__ == '__main__':
    process_csv(
        input_path='data_preproccess/cleaned_data.csv',
        output_normalized='data/target.csv',
        output_no_tone='data/source.csv'
    )
