import re
import os
from dotenv import load_dotenv
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


load_dotenv() # Lädt die Umgebungsvariablen aus der .env Datei
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Holt den OpenAI API Key aus den Umgebungsvariablen


def get_transcript(video_url: str) -> str:
    video_id = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", video_url).group(1) # Extrahiert die Video-ID aus der URL
    ytt_api = YouTubeTranscriptApi() # Initialisiert die YouTube Transcript API
    transcript = next(iter(ytt_api.list(video_id))).fetch() # Holt das Transkript für die Video-ID
    text = " ".join([entry.text for entry in transcript]) # Kombiniert alle Textteile des Transkripts zu einem einzigen String
    return text 


def chunk_text(text):
    doc = Document(text=text) # Erstellt ein Dokument-Objekt aus dem Text
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50) # Initialisiert ein SentenceSplitter mit einer Chunk-Größe von 512 und einer Überlappung von 50 damit beim Retrieval Kontext erhalten bleibt
    nodes = splitter.get_nodes_from_documents([doc]) # Teilt das Dokument in kleinere Chunks auf
    return nodes


def embed_texts(nodes):
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", 
                                  api_key=OPENAI_API_KEY) # Initialisiert das Einbettungsmodell
    index = VectorStoreIndex(nodes, embed_model=embed_model) # Erstellt einen Vektor-Index in dem die Chunks gespeichert werden, ermöglicht das Retrieval von relevanten Chunks. Wird auf Basis des Einbettungsmodells erstellt
    return index


def answer_question(index, question):
    llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY) # Initialisiert das LLM
    query_engine = index.as_query_engine(llm=llm) # Erstellt eine Query-Engine auf Basis der Wissensbasis
    response = query_engine.query(question) # Stellt die Frage an die Query-Engine. beinhlatet Retrieval und Generierung der Antwort
    return response


def use_rag(yt_video_url: str, question: str):
    try:
        transcript = get_transcript(yt_video_url)
        nodes = chunk_text(transcript) 
        index = embed_texts(nodes)
        response = answer_question(index, question)
        return response
    except Exception as e:
        return f"Fehler beim Verarbeiten: {str(e)}"


if __name__ == "__main__":
    
    if not OPENAI_API_KEY:
        raise RuntimeError("Kein API Key gefunden! Prüfe .env Datei")

    url = "https://www.youtube.com/watch?v=uTwRvAM682c"
    question = "Wann darf ich durch eine Anliegerstraße fahren?"

    response = use_rag(url, question)
    print("Antwort:")
    print(response)