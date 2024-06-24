import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd

# Thiết lập đường dẫn đến Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'G:\Tesseract OCR\tesseract.exe' # Thay đổi đường dẫn theo hệ thống của bạn

# Hàm chuyển đổi hình ảnh thành văn bản
def image_to_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='vie')  # 'vie' là mã ngôn ngữ cho tiếng Việt
    return text

# Tải mô hình và tokenizer PhoBERT
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Tạo pipeline cho NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Đường dẫn đến hình ảnh đơn thuốc
image_path = r'C:\Users\USER\Desktop\PrescriptionProject_DBM301\custom\images\train\prescription_page-0015.jpg'

# OCR: Chuyển đổi hình ảnh thành văn bản
text = image_to_text(image_path)
print("Text from image:", text)

# NER: Phân tích văn bản để nhận diện tên thuốc
ner_results = ner_pipeline(text)

# Hiển thị kết quả
recognized_medications = []
for entity in ner_results:
    if entity['entity'] == 'I-DRUG':
        recognized_medications.append(entity['word'])

if recognized_medications:
    print("Recognized medications:")
    for med in recognized_medications:
        print(med)
else:
    print("Không tìm thấy tên thuốc nào được nhận diện.")

# Load danh sách tên thuốc từ file Excel
file_path = r'C:\Users\USER\Desktop\PrescriptionProject_DBM301\prepare\data_tenthuoc.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Kiểm tra các tên thuốc đã nhận diện có trong danh sách hay không
matched_medications = set(recognized_medications) & set(df['TÊN THUỐC'].tolist())
print("Các thuốc được nhận diện và khớp trong danh sách:")
for med in matched_medications:
    print(med)
