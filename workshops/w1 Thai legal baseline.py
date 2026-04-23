# dki9yf8jk Tokenize + custom_Dict
import re
LEGAL_KEYWORD = ["มาตรา","พ.ร.บ.","คำพิพากษา","ศาลฎีกา","ศาลอุทธรณ์","ศาลชั้นต้น"]

def legal_tokenizer(text):
    # ใช้ Regex จัดการเบ้องต้น ค่อยตัดส่วนที่เหลือ
    compound = "|".join(map(re.escape,sorted(LEGAL_KEYWORD,key = len, reverse=True)))
    pattern = (compound + r"|[\u0E00-\u0E7F]+" + r"|[A-Za-z0-9]+")
    return re.findall(pattern, text)



test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรเเละเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)
print(f"Input: {test_text}")
print(f"Output: {tokens}")
   
# ตัดด้วย Deep Learning (Deepcut)  
# import deepcut
# print(f"output Deepcut: {deepcut.tokenize(test_text)}")

#ส่งงานเเเเkljlkj;k;lkll
