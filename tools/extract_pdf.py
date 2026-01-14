"""
读取PDF文档并提取文本内容
"""
import sys
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    print("PyPDF2未安装，尝试使用其他方法...")
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber也未安装，请安装PDF读取库：")
        print("pip install PyPDF2 或 pip install pdfplumber")
        sys.exit(1)

def extract_pdf_text(pdf_path):
    """提取PDF文本内容"""
    text = []
    
    # 尝试使用PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text.append(f"=== Page {page_num + 1} ===\n")
                text.append(page.extract_text())
                text.append("\n\n")
        return ''.join(text)
    except Exception as e:
        print(f"PyPDF2读取失败: {e}")
    
    # 尝试使用pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text.append(f"=== Page {page_num + 1} ===\n")
                text.append(page.extract_text())
                text.append("\n\n")
        return ''.join(text)
    except Exception as e:
        print(f"pdfplumber读取失败: {e}")
        return None

if __name__ == "__main__":
    pdf_path = Path(__file__).parent.parent / "2505.15155v2.pdf"
    
    print(f"正在读取PDF文件: {pdf_path}")
    print(f"文件大小: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    text = extract_pdf_text(pdf_path)
    
    if text:
        output_path = Path(__file__).parent.parent / "2505.15155v2_extracted.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"文本已提取到: {output_path}")
        print(f"总字符数: {len(text)}")
    else:
        print("提取失败")
        sys.exit(1)
