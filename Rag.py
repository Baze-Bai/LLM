import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from Store_data import store_data
import os
import requests
import json
import streamlit as st
from dotenv import load_dotenv


def main():
    # 缓存 API key 的加载
    @st.cache_data
    def get_bearer():
        load_dotenv()
        return 'Bearer ' + os.getenv("DEEP_SEEK_API_KEY")

    # 缓存数据的加载（假设 store_data() 返回 chunks 和 index）
    @st.cache_resource
    def load_chunks_and_index():
        return store_data()  # 返回 chunks, index

    # 缓存模型的加载
    @st.cache_resource
    def load_model():
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 主逻辑中调用缓存函数
    Bear = get_bearer()
    chunks, index = load_chunks_and_index()
    model = load_model()

    # 示例用户问题
    st.title("Battery Knowledge Rag")

    # 1. 用户输入问题
    question = st.text_input("Enter your question here:")

    # 可以根据需要添加更多选项，如温度、top_k 等

    # 2. 当用户点击按钮时，触发 RAG + 流式调用
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

    # 将问题编码为向量（并归一化）
    q_embedding = model.encode([question])
    q_embedding = np.array(q_embedding, dtype="float32")
    faiss.normalize_L2(q_embedding)

    
    # 在FAISS中检索与问题最相关的5个文本片段
    k = 5
    D, I = index.search(q_embedding, k)  # D是相似度分数，I是索引
    top_idx = I[0]    # 最相关的文本片段索引列表
    top_chunks = [chunks[i] for i in top_idx]


    context_text = "\n".join(top_chunks)  # 将多个片段用换行连接

    prompt = f"Answer the user's questions based on the following documentation。\ndocument content:\n{context_text}\n\nquestion: {question}\n"


    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
            "model": "deepseek-ai/DeepSeek-V3", # 替换成你的模型
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": True, # 此处需要设置为stream模式
            "temperature": 1.3,
            "frequency_penalty": 0.65
    }

    headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": Bear
        }

    # 2.4 发起流式请求并实时显示返回
    st.write("**Answer:**")
    answer_container = st.empty()  # 用于实时更新答案的容器
    answer_text = ""  # 累积的回答内容
    try:
            response = requests.post(url, json=payload, headers=headers, stream=True)
            if response.status_code == 200:
                # 用于跳出循环的标志
                done = False
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8', errors='ignore')
                        for line in decoded_chunk.split('\n'):
                            line = line.strip()
                            if line.startswith('data: '):
                                json_str = line[6:]  # 移除 "data: "
                                if json_str == '[DONE]':
                                    done = True
                                    break
                                try:
                                    data = json.loads(json_str)
                                    content = data['choices'][0]['delta'].get('content', '')
                                    # 累加回答文本
                                    answer_text += content
                                    # 实时更新到页面
                                    answer_container.markdown(answer_text)
                                except json.JSONDecodeError:
                                    # 如果解析失败，可以选择打印或忽略
                                    print(f"JSON parse error: {json_str}")
                        if done:
                            break
            else:
                st.error(f"Request failed with status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    # 在本地运行: streamlit run app.py
    main()