
FROM python:3.12-slim

WORKDIR /Battery_Rag

COPY requirements.txt /Battery_Rag/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /Battery_Rag

EXPOSE 8501

CMD ["streamlit", "run", "Rag.py", "--server.port=8501", "--server.address=0.0.0.0"]