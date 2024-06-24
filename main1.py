import pytesseract
from PIL import Image
import spacy
from drug_named_entity_recognition import find_drugs

# Thiết lập đường dẫn đến Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'G:\Tesseract OCR\tesseract.exe'

# Hàm chuyển đổi hình ảnh thành văn bản
def image_to_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='vie')  # 'vie' là mã ngôn ngữ cho tiếng Việt
    return text

# Hàm loại bỏ các tên thuốc trùng lặp
def remove_duplicate_drugs(ner_results):
    unique_drugs = []
    seen_drugs = set()

    for result in ner_results:
        drug_name = result[0]['name']
        if drug_name not in seen_drugs:
            unique_drugs.append(result)
            seen_drugs.add(drug_name)

    return unique_drugs

# Đường dẫn đến hình ảnh đơn thuốc
image_path = r'C:\Users\USER\Desktop\PrescriptionProject_DBM301\custom\images\train\prescription_page-0050.jpg'

# OCR: Chuyển đổi hình ảnh thành văn bản
text_from_image = image_to_text(image_path)
print("Văn bản từ hình ảnh:")
print(text_from_image)

# Sử dụng Spacy để phân tích văn bản bằng mô hình đã được huấn luyện trước (en_core_web_sm)
nlp = spacy.load("en_core_web_sm")
doc = nlp(text_from_image)

# Trích xuất các token từ văn bản đã phân tích
tokens = [token.text for token in doc]

# Sử dụng find_drugs để nhận diện tên thuốc từ các token
ner_results = find_drugs(tokens, is_ignore_case=True)

# Loại bỏ các tên thuốc trùng lặp
unique_drugs = remove_duplicate_drugs(ner_results)

# Hiển thị tên thuốc được nhận diện
if unique_drugs:
    print("\nCác thuốc được nhận diện:")
    for med in unique_drugs:
        print(med[0]['name'])  # Truy cập từ điển trong tuple và sau đó truy cập 'name'
else:
    print("\nKhông tìm thấy tên thuốc nào được nhận diện.")
