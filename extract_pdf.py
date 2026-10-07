import PyPDF2
import sys

sys.stdout.reconfigure(encoding='utf-8')

pdf_path = 'PIIS2589004225004833.pdf'

try:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        print(f"总页数: {total_pages}\n")
        
        for i in range(min(5, total_pages)):
            print(f"=== 第 {i+1} 页 ===\n")
            text = pdf_reader.pages[i].extract_text()
            print(text)
            print("\n")
except Exception as e:
    print(f"错误: {e}")
