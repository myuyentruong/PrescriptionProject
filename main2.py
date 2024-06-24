import pytesseract
from PIL import Image
import spacy
import pandas as pd
from drug_named_entity_recognition import find_drugs

# Thiết lập đường dẫn đến Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'G:\Tesseract OCR\tesseract.exe'

# Hàm chuyển đổi hình ảnh thành văn bản
def image_to_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='vie')  # 'vie' là mã ngôn ngữ cho tiếng Việt
    return text

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    # Xóa dấu "-" và "_"
    text = text.replace('-', '').replace('_', '')
    # Thay thế "0" thành "o"
    text = text.replace('0', 'o')
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
image_path = r'C:\Users\USER\Desktop\PrescriptionProject_DBM301\custom\images\train\prescription_page-0004.jpg'

# OCR: Chuyển đổi hình ảnh thành văn bản
text_from_image = image_to_text(image_path)
print("Văn bản từ hình ảnh:")
print(text_from_image)

# Tiền xử lý văn bản
processed_text = preprocess_text(text_from_image)
print("\nVăn bản sau khi tiền xử lý:")
print(processed_text)

# Sử dụng Spacy để phân tích văn bản bằng mô hình đã được huấn luyện trước (en_core_web_sm)
nlp = spacy.load("en_core_web_sm")
doc = nlp(processed_text)

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

    # Load file Excel vào DataFrame của pandas
    excel_file = r'C:\Users\USER\Desktop\PrescriptionProject_DBM301\prepare\data_tenthuoc.xlsx'  # Thay đổi đường dẫn đến file Excel của bạn
    df = pd.read_excel(excel_file, sheet_name='Sheet1')  # Đổi tên sheet nếu cần

    # Duyệt qua từng tên thuốc được nhận diện và tìm kiếm trong DataFrame
    for med in unique_drugs:
        drug_name = med[0]['name']
        matching_rows = df[df['TÊN THUỐC'].str.contains(drug_name, case=False, na=False)]

        if not matching_rows.empty:
            print(f"\nThông tin cho thuốc '{drug_name}':")
            for index, row in matching_rows.iterrows():
                print(f"Tên thuốc: {row['TÊN THUỐC']}")
                print(f"- Công dụng: {row['CÔNG DỤNG']}")  
                print(f"- Tác dụng phụ: \n{row['TÁC DỤNG PHỤ']}") 
                print(f"- Lưu ý: \n{row['LƯU Ý']}") 
                print(f"- Dị ứng/Chống chỉ định: \n{row['DỊ ỨNG / CHỐNG CHỈ ĐỊNH']}") 

else:
    print("\nKhông tìm thấy tên thuốc nào được nhận diện.")
