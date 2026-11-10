from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import re
import jionlp as jio



#1.配置模型
embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url="http://192.168.1.105:11434")

#2.加载文档
def word_load(word_path):
    word_loader = UnstructuredWordDocumentLoader(word_path,mode="elements")

    docs = word_loader.load()
    return docs

#3.清洗文档
#3.1正则化匹配清洗
def clean_pdf_text_re(text: str) -> str:
    _SECTION_HEADERS = "提示信息|警告！|注意！|说明！|帮助信息|图标和插图说明"
    """清洗 PDF 提取文本中的排版折行，保留段落/标题结构。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 合并汉字之间的排版折行（如 造\n成、版\n本）
    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)

    # 合并标点后的折行（如 部位，\n其他）
    text = re.sub(r"(?<=[，。；：！？、）」』])\n(?=[\u4e00-\u9fff])", "", text)

    # 在常见标题词后恢复换行，避免「提示信息警告！」粘在一起
    text = re.sub(rf"({_SECTION_HEADERS})(?=[\u4e00-\u9fff])", r"\1\n", text)

    # 多个空行压成段落分隔
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

#3.2构建清洗后的Document
def create_cleaned_documents(documents):
    cleaned_documents = []
    for doc in documents[50:100]:
        cleaned_content = clean_pdf_text_re(doc.page_content)
        cleaned_content = jio.clean_text(cleaned_content)
        
        cleaned_document = Document(
            page_content=cleaned_content,
            metadata = doc.metadata
        )

        cleaned_documents.append(cleaned_document)
    return cleaned_documents
    
#4. 切割文档 + 过滤文档
def creat_chunks(cleaned_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        # 设置分隔符，这里使用 ".!?" 来匹配句子结束符号
        separators=["\n\n", "\n", "(?<=\。！？ )", " ", ""],
        # 设置每个片段的最大长度 chunk_size 一般根据文档内容或大小来设置,
        chunk_size=100,
        # 设置每个片段之间的重叠字符数量   chunk_overlap = chunk_size * 20% 或 30%
        chunk_overlap=15,
        # 设置是否长内容优先
        length_function=len
    )

    chunks_list = text_splitter.split_documents(cleaned_documents)

    #5.过滤文档
    chunks = [c for c in chunks_list if len(c.page_content)>30]
    
    return chunks

#6.保存到向量库
# def create_db(chunks):
#     db = Chroma.from_documents(
#         documents=chunks,
#         collection_name="car_info_2",
#         embedding=embedding_model,
#         persist_directory=r"E:\PythonProjects\2026.10.27 lang_chain\20261109 RAG_1\data\chroma_db"
#     )
#     return db

#6.2追加向量库
def add_db(new_chunks):
    db = Chroma(
        collection_name="car_info_2",
        embedding_function=embedding_model,
        persist_directory=r"E:\PythonProjects\2026.10.27 lang_chain\20261109 RAG_1\data\chroma_db" 
    )
    db.add_documents(new_chunks)
    
    print(f"成功追加了 {len(new_chunks)} 条数据！")
    return db
    
#7. 主函数
def main(word_path):
    documents = word_load(word_path)
    cleaned_documents = create_cleaned_documents(documents)
    chunks = creat_chunks(cleaned_documents)
    db = add_db(chunks)
    return db
if __name__ == "__main__":
    word_path = r"E:\PythonProjects\2026.10.27 lang_chain\20261109 RAG_1\data\峨眉山讲解词.docx"
    final_db = main(word_path)