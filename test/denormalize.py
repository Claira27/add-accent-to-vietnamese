import re

def denormalize_tone_line(normalized_str):
    # Danh sách ký hiệu không dấu + số (đầu ra của normalize_tone_line)
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

    # Danh sách ký tự có dấu (đầu vào của normalize_tone_line)
    intab_l = "áàảãạâấầẩẫậăắằẳẵặđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    intab_u = "ÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
    intab = list(intab_l + intab_u)

    # Tạo từ điển ánh xạ ngược (từ ký hiệu không dấu + số về ký tự có dấu)
    replaces_dict = dict(zip(outtab, intab))

    # Sử dụng regex để thay thế, ưu tiên các ký hiệu dài hơn trước
    pattern = '|'.join(re.escape(k) for k in sorted(replaces_dict.keys(), key=len, reverse=True))
    return re.sub(pattern, lambda m: replaces_dict[m.group(0)], normalized_str)

# Kiểm tra cả hai hàm
def normalize_tone_line(utf8_str):
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

# Kiểm tra
# test_str = "tôi yêu việt nam"
# normalized = normalize_tone_line(test_str)
# denormalized = denormalize_tone_line(normalized)
# print(f"Original: {test_str}")
# print(f"Normalized: {normalized}")
# print(f"Denormalized: {denormalized}")