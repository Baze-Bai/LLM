DataBase: MongoDB
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
LLM: DeepSeek-V3
Chunking method: Chunking according to titles first then slice them into chunks < 256 tokens
Similarity Calculation: Using cosine to calculate the similarity between different vector embeddings

This rag is used to answer the users' questions about battery, if the questions are not related to battery, the rag will choose to refuse answer.