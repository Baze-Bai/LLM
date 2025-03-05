from Store_data import store_data
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from Extact_PDF import extract_text_from_pdf, text_chunk
import os


chunks, index= store_data()
# 将文本块存入 MongoDB（或PostgreSQL等关系库）
client = MongoClient("mongodb://localhost:27017/")
db = client["rag_app"]
col = db["pdf_chunks"]
# 清空集合再插入数据
col.delete_many({})
docs = [{"chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks)]
col.insert_many(docs)
print("已保存文本块数量到数据库：", col.count_documents({}))