import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq  # ✅ Official Groq LLM client
from dotenv import load_dotenv
import sqlite3
from langchain_core.tools import tool
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.documents import Document

class RelevanceOutput(BaseModel):
    is_relevant: bool

class refinedQuery(BaseModel):
    refined_query: str

class queryExecuteStructure(BaseModel):
    refined_query: str

load_dotenv()
# Load your RAG schema
with open("rag_schema.json") as f:
    schema = json.load(f)

# Convert schema into LangChain Documents
def schema_to_docs(schema):
    docs = []
    for table, content in schema.items():
        desc = content.get("description", "")
        doc = f"Table: {table}\nDescription: {desc}\nColumns:"
        for col in content["columns"]:
            doc += f"\n- {col['name']} ({col['type']}): {col.get('description', '')}"
        docs.append(Document(page_content=doc))
    return docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # Safe limit for Groq / Claude / OpenAI
    chunk_overlap=100     # Helps with edge continuity
)

docs = schema_to_docs(schema)
split_docs = text_splitter.split_documents(docs)


# Sentence Transformers for embedding
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Store: Qdrant
faiss_index = FAISS.from_documents(
    split_docs,
    embedding=embedding_model
)

retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt template
parser = PydanticOutputParser(pydantic_object=queryExecuteStructure)
prompt_template = PromptTemplate.from_template(f"""
You are a helpful HR assistant that understands database schema and writes SQL queries.

Based on the schema below:
{{context}}

Translate this question into SQL only is the questions is to read data , if any other operation is intendd , do not generate any qyery:
Question: {{question}}
                                               
Make sure to only create for read operation do not create query for any other operation, any other crud operation should not be allowed and mark the same                                              
SQL:
{parser.get_format_instructions()}
""")

# ✅ Groq LLM via langchain_groq
llm = ChatGroq(model="gemma2-9b-it", temperature=0)

# LangChain chain setup
rag_chain = prompt_template | llm | parser

# Final query function
def ask_sql_question(question: str):
    schema_docs = retriever.get_relevant_documents(question)
    schema_text = "\n\n".join(doc.page_content for doc in schema_docs)
    response = rag_chain.run({"context": schema_text, "question": question})
    response = response.dict()

    queryResponse = executeAgentQuery(response['refined_query'])



def executeAgentQuery(sql_query:str) -> dict:
    """Returns the data for the generated query by executing it in the database"""
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect("hr_system.db")
        conn.row_factory = sqlite3.Row  # allows access to columns by name
        cursor = conn.cursor()

        # Execute the query
        cursor.execute(sql_query)
        rows = cursor.fetchall()

        # Get column names
        column_names = [col[0] for col in cursor.description]

        # Convert to list of dicts
        result = [dict(zip(column_names, row)) for row in rows]

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        conn.close()

sql_generation_tool = Tool(
    name="generate_sql",
    description="Generates SQL queries from natural language using schema-based RAG",
    func=ask_sql_question
)


def checkQueryRelevance(query):

    parser = PydanticOutputParser(pydantic_object=RelevanceOutput)
    relevance_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""You are an assistant responsible for filtering HR-related SQL tasks.

        Only consider a question relevant if:
        - It is about **attendance** or **leave management**
        - It is meant to **read (SELECT)** or **create (INSERT)** information

        If the question involves **delete, update, edit, or modify**, or is about anything outside attendance/leaves (e.g. payroll, personal info), mark it as **not relevant**.

        Respond ONLY in the following JSON format:

        {parser.get_format_instructions()}

        Question: {{question}}
        """
            )
        ])
    

    relevance_chain = relevance_prompt | llm | parser
    response = relevance_chain.invoke({"question":query})
    return response.dict()

class AgentState(TypedDict, total=False):
    user_id: int
    question: HumanMessage                      # raw user input as message
    messages: List[BaseMessage]                 # full chat history
    refined_question: str                       # output from refiner node
    is_attendance_marked: bool                  # backend logic result
    is_relevant: bool                           # output from filter
    documents: List[Document]                   # retrieved context
    sql_query: str                              # generated SQL
    query_result: dict                          # result from DB
    response: str                               # final LLM answer
    rephrase_count: int                         # how many refinement attempts
    proceed_to_generate: bool                   # gate control


def refineQuestion(state : AgentState) -> AgentState:
    
    parser = PydanticOutputParser(pydantic_object=refinedQuery)
    refine_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an assistant that rewrites user questions into clear, self-contained queries using chat history and context.

    Use the provided memory and user state to make vague or short queries clearer.

    Guidelines:
    - Preserve original intent
    - Do not hallucinate new facts
    - Output only the refined question (no preamble)

    Chat History:
    {chat_history}

    Attendance marked today: {is_attendance_marked}

    Original Question:
    {question}

    {parser.get_format_instructions()}
    """
        )
    ])


    refine_chain = refine_prompt | llm | parser
    response = refine_chain.invoke({"chat_history":state["messages"],"is_attendance_marked":"true","question":state["question"]})
    return response.dict()




from langgraph.graph import END, StateGraph, START

def getRelevance(state : AgentState)-> AgentState:  

    graphState = state
    getQueryRelevance = checkQueryRelevance(state['question'])

    print(getQueryRelevance)

    

