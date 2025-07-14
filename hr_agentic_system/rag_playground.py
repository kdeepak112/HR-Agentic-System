import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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
from langchain_core.messages import BaseMessage, HumanMessage , AIMessage
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
import warnings
warnings.filterwarnings('ignore')


load_dotenv()

class RelevanceOutput(BaseModel):
    is_relevant: bool

class refinedQuery(BaseModel):
    refined_query: str

class queryExecuteStructure(BaseModel):
    refined_query: str

class queryAnswerStructure(BaseModel):
    answer: str

class AgentState(TypedDict, total=False):
    question: HumanMessage                      # raw user input as message
    messages: List[BaseMessage]                 # full chat history
    refined_question: str                       # output from refiner node                 # backend logic result
    is_relevant: bool                           # output from filter                # retrieved context
    sql_query: str                              # generated SQL
    query_result: dict                          # result from DB
    response: str                               # final LLM answer



# ✅ Groq LLM via langchain_groq
llm = ChatGroq(model="gemma2-9b-it", temperature=0)
checkpointer = MemorySaver()

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


# Final query function
def ask_sql_question(question: str):

    # Prompt template
    parser = PydanticOutputParser(pydantic_object=queryExecuteStructure)
    format_instructions = parser.get_format_instructions()

    prompt_template = PromptTemplate.from_template(f"""
    You are a helpful HR assistant that understands database schema and writes SQL queries.

    Based on the schema below:
    {{context}}

    Translate this question into SQL only is the questions is to read data , if any other operation is intendd , do not generate any qyery:
    Question: {{question}}
                                                
    Make sure to only create for read operation do not create query for any other operation, any other crud operation should not be allowed and mark the same                                              
    SQL:
    Respond ONLY in the following JSON format:
        {{format_instructions}}
    """)



    # LangChain chain setup
    rag_chain = prompt_template | llm 

    schema_docs = retriever.get_relevant_documents(question)
    schema_text = "\n\n".join(doc.page_content for doc in schema_docs)
    response = rag_chain.invoke({"context": schema_text, "question": question,"format_instructions":format_instructions})
    aimessage = response

    parser_response = parser.invoke(response).dict()

    queryResponse = executeAgentQuery(parser_response['refined_query'])

    return (aimessage,queryResponse,parser_response)

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

def checkQueryRelevance(query):

    parser = PydanticOutputParser(pydantic_object=RelevanceOutput)
    format_instructions = parser.get_format_instructions()

    relevance_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"""You are an assistant responsible for filtering HR-related SQL tasks.

        Consider a question **relevant** if:
        - It is related to **employees**, including general employee data like headcount, names, roles, departments, etc.
        - OR it is about **attendance** or **leave management**
        - AND it is meant to **read (SELECT)** or **create (INSERT)** information

        Consider a question **not relevant** if:
        - It involves **update, edit, delete, or modify** operations
        - OR it is about unrelated areas such as **payroll**, **salary**, **personal information**, or any **non-employee topic**


        Respond ONLY in the following JSON format:
        {{format_instructions}}

        Question: {{question}}
        """
            )
        ])
    

    relevance_chain = relevance_prompt | llm | parser
    response = relevance_chain.invoke({"question":query,"format_instructions":format_instructions})
    print(response)
    return response.dict()

def refineQuestion(messages , question):
    
    parser = PydanticOutputParser(pydantic_object=refinedQuery)
    format_instructions = parser.get_format_instructions()

    refine_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""
                You are an assistant that rewrites user questions into clear, self-contained queries using chat history and context.

                Use the provided memory and user state to make vague or short queries clearer.

                Guidelines:
                - Preserve original intent
                - Do not hallucinate new facts
                - Output only the refined question (no preamble)

                Chat History:
                {{chat_history}}

                Respond ONLY in the following JSON format:
                {{format_instructions}}

                Original Question:
                {{question}}

                
            """
        )
    ])


    refine_chain = refine_prompt | llm 
    response = refine_chain.invoke({"chat_history":messages,"question":question,"format_instructions":format_instructions})
    aimessage = response
    parsedoutput = parser.invoke(response)
    return (parsedoutput.dict() , aimessage)

def generateAnswer(question,sql_query,query_output):
    parser = PydanticOutputParser(pydantic_object=queryAnswerStructure)
    format_instructions = parser.get_format_instructions()

    answerPrompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""

            you are being provided with the question asked by the employee , the sql query generated for it , and the output of the query in json format , you need to create an answer
            for sending to the user.

            question : {{question}}
            sql_query : {{sql_query}}
            query_output : {{query_output}}

            Respond ONLY in the following JSON format:
            {{format_instructions}}


            """
        )
    ])
    
    answer_chain = answerPrompt | llm 
    response = answer_chain.invoke({"question":question,"sql_query":sql_query,"query_output":query_output,"format_instructions":format_instructions})
    aimessage = response 
    parsed_output = parser.invoke(response).dict()
    return (aimessage,parsed_output)




def getRelevance(state : AgentState)-> AgentState:  
    print(state)
    getQueryRelevance = checkQueryRelevance(state['question'])
    state['is_relevant'] = getQueryRelevance['is_relevant']

    return state

def getRefinedQuestion(state : AgentState)-> AgentState:  
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    refined_question , aimessage = refineQuestion(state['messages'],state['question'])
    state['refined_question'] = refined_question['refined_query']
    state['messages'].append(aimessage)
    return state

def get_sql_query(state : AgentState)-> AgentState:  

    aimessage , queryResponse , sql_query = ask_sql_question(state['refined_question'])
    state['query_result'] = queryResponse
    state['sql_query'] = sql_query
    state['messages'].append(aimessage)
    return state 

def get_answer(state: AgentState) -> AgentState:
    aimessage, generated_answer = generateAnswer(
        state['refined_question'],
        state['sql_query'],  # But you're not storing this yet!
        state['query_result']
    )
    state['response'] = generated_answer['answer']
    state['messages'].append(aimessage)
    return state

def check_relevance(state: AgentState)-> AgentState:
    if state['is_relevant'] :
        return 'getRefinedQuestion'
    else:
        return 'cannot_answer'

def cannot_answer(state: AgentState):
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(
            content="I'm sorry, but I cannot find the information you're looking for."
        )
    )
    return state


# Workflow
workflow = StateGraph(AgentState)
workflow.add_node("getRelevance", getRelevance)
workflow.add_node("getRefinedQuestion", getRefinedQuestion)
workflow.add_node("get_sql_query", get_sql_query)
workflow.add_node("get_answer", get_answer)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("cannot_answer", cannot_answer)


workflow.add_conditional_edges(
    "getRelevance",
    check_relevance,
    {
        "getRefinedQuestion": "getRefinedQuestion",
        "cannot_answer": "cannot_answer",
    },
)
workflow.add_edge("getRefinedQuestion", "get_sql_query")
workflow.add_edge("get_sql_query", "get_answer")
workflow.add_edge("get_answer", END)
workflow.add_edge("cannot_answer", END)

workflow.set_entry_point("getRelevance")
graph = workflow.compile(checkpointer=checkpointer)


input_data = {"question": HumanMessage(content="How many employees in the system ")}
response = graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})
    
print(response)
