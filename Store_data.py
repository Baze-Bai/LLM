import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from Extact_PDF import extract_text_from_pdf, text_chunk
import os

def store_data():
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path = os.path.join(current_dir, 'battery_ppt')

    text = extract_text_from_pdf(folder_path)
    chunk_dict, chunks = text_chunk(text)

    # 加载预训练的句向量模型（MiniLM 示例）
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    # 计算每个文本块的向量表示
    embeddings = model.encode(chunks)  # shape: (num_chunks, 384)
    embeddings = np.array(embeddings, dtype="float32")

    # 创建索引对象，构建 FAISS 索引（使用余弦相似度，将向量先归一化然后用内积检索）
    vec_dim = embeddings.shape[1]  # 向量维度，例如 all-MiniLM-L6-v2 为384
    index = faiss.IndexFlatIP(vec_dim)
    # 若使用余弦相似，需要先对向量进行L2归一化
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    # Store the index to disk
    faiss.write_index(index, "my_faiss.index")
    
    return chunks, index
