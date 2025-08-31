import os
import asyncio
import requests
from typing import Dict, List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tabulate import tabulate
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool, StructuredTool
from langchain.tools.render import render_text_description
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag import use_rag

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)


## RAG Tool

class RagInput(BaseModel): # Input Schema für das RAG Tool, definiert die erwarteten Felder
    yt_video_url: str = Field(
        ...,
        description="Vollständige YouTube-URL oder Video-ID des Videos, zu dem eine Frage gestellt wird.",
    ),
    question: str = Field(
        ...,
        description="Die Frage zum Inhalt des Videos.",
    )

def rag(yt_video_url: str, question: str) -> str: # Wrapper-Funktion für RAG, fügt Logging hinzu
    print(f"RAG gestartet für Video: {yt_video_url} mit Frage: {question}")
    return use_rag(yt_video_url, question)

video_rag_tool = StructuredTool.from_function( # Erstellt ein StructuredTool aus der RAG-Funktion
    func=use_rag,
    name="Video-RAG",
    description=(
        "Nutze dieses Tool IMMER, wenn der Nutzer eine Frage zum Inhalt eines YouTube-Videos stellt. "
        "Input: yt_video_url (vollständige YouTube-URL oder Video-ID) und question (Frage zum Video). "
        "Output: Eine Antwort basierend auf dem Transkript."
    ),
    args_schema=RagInput,
)


## Summarization Tool

def summarize(text: str) -> str: # Funktion zur Textzusammenfassung
    prompt = (
        f"Fasse den folgenden Text zusammen: {text}"
    )
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return f"[Fehler bei der Zusammenfassung: {e}]"

# Basic LLM Tool

def llm_tool_func(query: str) -> str: # Funktion zur Beantwortung allgemeiner Anfragen durch das LLM
    prompt = f"Beantworte die folgende Anfrage so gut wie möglich: {query}"
    try:
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        return f"[Fehler bei der LLM-Antwort: {e}]"

llm_tool = StructuredTool.from_function( # Erstellt ein StructuredTool aus der LLM-Funktion
    func=llm_tool_func,
    name="law_tool",
    description="Beantwortet Anfragen so gut wie möglich.",
)

## Law Case Tool

class CaseSearchInput(BaseModel): # Input Schema für das Law Case Tool, definiert die erwarteten Felder
    id: str = Field(
        ...,
        description="Numerische Case-ID nach gesucht werden soll.",
    )

def search_cases(id: str) -> str: # Funktion zur Suche von Gerichtsentscheidungen nach ID
    url = f"https://de.openlegaldata.io/api/cases/{id}"
    resp = requests.get(url) # Sendet eine GET-Anfrage an die API
    resp.raise_for_status()

    data = resp.json()

    court = data['court']['name']
    date = data['created_date']
    type = data['type']
    summary = summarize(data['content']) # Fasst den Inhalt der Entscheidung zusammen

    return f"Datum: {date} | Gericht: {court} | Typ: {type} | Zusammenfassung: {summary}" # Formatiert die Ausgabe

law_case_tool = StructuredTool.from_function( # Erstellt ein StructuredTool aus der Case Search Funktion
    func=search_cases,
    name="law_case_search",
    description="Sucht Gerichtsentscheidungen nach ID und gibt Datum, Gerichtsname, Entscheidungstyp und eine Zusammenfassung zurück.",
    args_schema=CaseSearchInput,
)


## Artwort-Schema

class Decision(BaseModel): # Modell zur Entscheidung, ob die Antwort als Text oder Tabelle formatiert werden soll
    mode: Literal["text", "table"]

class TableAnswer(BaseModel): # Modell für tabellarische Antworten
    type: Literal["table"] = "table"
    caption: str 
    columns: List[str]
    rows: List[List[str]]


def final_response(response_dict, user_input) -> str: # Generiert die finale Antwort basierend auf der Entscheidung, entscheidet zwischen Text und Tabelle
    
    answer = str(response_dict)

    decide_prompt = ChatPromptTemplate.from_messages([ # Prompt zur Entscheidung über das Antwortformat
        ("system", "Entscheide, ob die Antwort als Fließtext (wähle 'text') oder als Tabelle (wähle 'table') ausgegeben werden soll. Beachte das nur bei ähnlichen Teilanfragen in einer Tabelle geantwortet werden sollen. Sonst lieber Text nutzen."),
        ("user", "Orginal Input: {user_input}\nLLM Anwort: {answer}")
    ])
    decision_chain = decide_prompt | llm.with_structured_output(Decision) # Kette zur Entscheidung
    decision = decision_chain.invoke({"answer": answer, "user_input": user_input}) # Führt die Kette aus und erhält die Entscheidung
    
    if decision.mode == "table": # Wenn die Entscheidung "table" ist, generiere eine Tabelle
        table_prompt = ChatPromptTemplate.from_messages([
            ("system", "Erzeuge eine Tabelle + 1–2 Sätze caption."),
            ("user", "Orginal Input: {user_input}\nLLM Anwort: {answer}")
        ])
        table_chain = table_prompt | llm.with_structured_output(TableAnswer)
        table = table_chain.invoke({"answer": answer, "user_input": user_input})

        ascii_table = tabulate(table.rows, headers=table.columns, tablefmt="grid")
        return f"{table.caption}\n\n{ascii_table}"

    elif decision.mode == "text": # Wenn die Entscheidung "text" ist, generiere Fließtext
        text_prompt = ChatPromptTemplate.from_messages([
            ("system", "Erzeuge einen Humanen Fließtext als Antwort."),
            ("user", "Orginal Input: {user_input}\nLLM Anwort: {answer}")
        ])
        text_chain = text_prompt | llm
        text = text_chain.invoke({"answer": answer, "user_input": user_input})
        return text.content

## Paralell

tools = [video_rag_tool, law_case_tool, llm_tool] # Liste der verfügbaren Tools

tools_desc = render_text_description(tools) # Rendert die Beschreibung der Tools für den Agenten

agent_prompt = ChatPromptTemplate.from_messages([ # Prompt für den Agenten, der die Tools beschreibt und die Interaktion steuert
    ("system", "Du bist ein hilfreicher juristischer Assistent. Nutze Tools, wenn nötig."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
]).partial(tools= tools_desc)

agent = create_tool_calling_agent(llm, tools, agent_prompt) # Erstellt den Agenten mit Tool-Aufruf-Fähigkeiten ist das Herzstück des Agentensystems

agent_exec = AgentExecutor(agent=agent, tools=tools, verbose=False) # Executor für den Agenten, der die Ausführung der Agentenlogik und Tool-Interaktionen verwaltet

planner_prompt = ChatPromptTemplate.from_messages([ # Prompt zur Planung der Aufgabenzerlegung
    ("system", "Zerlege die Nutzeranfrage in atomare, unabhängige Teilaufgaben."),
    ("human", "{user_prompt}"),
    ("system", 'Gib NUR JSON-Liste zurück, z.B. ["Task A","Task B"].')
])
planner: Runnable = planner_prompt | llm | JsonOutputParser()

async def run_agent_parallel(user_prompt: str, max_concurrency: int = 5) -> Dict[str, str]: # Führt den Agenten parallel aus, um mehrere Teilaufgaben gleichzeitig zu bearbeiten
    tasks: List[str] = await planner.ainvoke({"user_prompt": user_prompt})
    async def call_one(task: str):
        agent_input = (
            f"Gesamte Nutzeranfrage:\n{user_prompt}\n\n"
            f"Teilaufgabe:\n{task}\n\n"
            "Wenn ein Tool erforderlich ist, wähle das passende und übergib alle benötigten Argumente."
        )
        res = await agent_exec.ainvoke({"input": agent_input}) # Führt den Agenten mit der Teilaufgabe aus
        return task, (res["output"] if isinstance(res, dict) and "output" in res else res)

    sem = asyncio.Semaphore(max_concurrency)
    async def limited(task: str):
        async with sem:
            return await call_one(task)

    results = await asyncio.gather(*(limited(t) for t in tasks))
    return dict(results)


## Main

if __name__ == "__main__":

    user_input = "Nutze das Rag Tool um den Inhalt des Videos (https://www.youtube.com/watch?v=uTwRvAM682c) zusammenzufassen? Gebe mir noch seperat an was ist Urteil Case-ID 346932?"
    #user_input = "Gebe mir die 4 Elemente der Erde, in einer struckturierten Darstellung."
    #user_input = "Suche nach Gerichtsurteil Case-ID: 346932 und 324590."


    response_dict = asyncio.run(run_agent_parallel(user_input, 5)) # Führt den Agenten mit der Nutzeranfrage aus, maximal 5 parallele Tasks
    
    print(f"### Anz. Teilaufgaben: {len(response_dict)} ###")
    output = final_response(response_dict, user_input) # Generiert die finale Antwort basierend auf den Teilergebnissen

    

    print(output)