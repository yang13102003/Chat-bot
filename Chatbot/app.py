import os
import chainlit as cl
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import openai
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from collections import deque
from literalai import LiteralClient
import nest_asyncio
from pathlib import Path
from uuid import uuid4
from collections import deque
import datetime
from typing import Optional
from flask import Flask, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
from typing import Optional
import asyncio

# Set API Keys securely
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")
os.environ["LITERAL_API_KEY"] = os.getenv("LITERAL_API_KEY")

# Initialize the Literal AI client
lai = LiteralClient(api_key=os.environ.get("LITERAL_API_KEY"))
lai.instrument_openai()

# Apply nest_asyncio to avoid issues with async loops in Jupyter and other environments
nest_asyncio.apply()

# Set environment variables for OAuth provider
os.environ['OAUTH_GOOGLE_CLIENT_ID'] = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
os.environ['OAUTH_GOOGLE_CLIENT_SECRET'] = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")

# Specify the path to the data folder
data_folder = 'data/Finance'

#Get the number of items in the data folder
def get_document_numbers(data_folder):
    # Convert the data_folder to a Path object
    data_folder_path = Path(data_folder)

    # Check if the data folder exists
    if not data_folder_path.exists():
        raise FileNotFoundError(f"The system cannot find the path specified: '{data_folder}'")
    
    # List all files in the data folder
    files = data_folder_path.iterdir()

    # Filter out only HTML files (not directories)
    html_files = [f for f in files if f.is_file() and f.suffix == '.html']

    # Generate a list of numbers based on the number of HTML files
    numbers = list(range(len(html_files)))
    return numbers

# Specify the numbers of the finance documents
numbers = get_document_numbers(data_folder)
# print(numbers) 

# Initialize UnstructuredReader and load the documents
loader = UnstructuredReader()
doc_set = {}
all_docs = []
for number in numbers:
    finance_docs = loader.load_data(
        file=Path(f"./data/Finance/Finance{number}.html"), split_documents=False
    )
    for d in finance_docs:
        d.metadata = {"number": number}
    doc_set[number] = finance_docs
    all_docs.extend(finance_docs)

# Print the first few documents to verify
# print(all_docs[:5])

# Initialize vector indices with chunk size
Settings.chunk_size = 512
index_set = {}
for number in numbers:
    storage_dir = Path(f"./storage/finance/{number}")
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Check if index already exists in storage
    if not any(storage_dir.iterdir()):
        # Create and persist index if it doesn't exist
        storage_context = StorageContext.from_defaults()
        cur_index = VectorStoreIndex.from_documents(
            doc_set[number],
            storage_context=storage_context,
        )
        index_set[number] = cur_index
        storage_context.persist(persist_dir=storage_dir)
    else:
        # Load existing index from storage
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index_set[number] = load_index_from_storage(storage_context)

# Create tools for each number index if not already created
individual_query_engine_tools = [
        QueryEngineTool(
            query_engine=index_set[number].as_query_engine(),
            metadata=ToolMetadata(
                name=f"vector_index_{number}",
                description=f"useful for when you want to answer queries about the {number} for finance documents",
            ),
        )
        for number in numbers
    ]

# Initialize query engine
query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
        llm=OpenAI(model="gpt-4o-mini"),
)

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description="useful for when you want to answer queries that require analyzing multiple Fianance documents",
    ),
)

tools = individual_query_engine_tools + [query_engine_tool]
agent = OpenAIAgent.from_tools(tools, verbose=True)

# Create message history for chat session
message_history = deque(maxlen=10)  # Keep the last 10 messages

@cl.on_chat_start
async def start():
    # Limit message history to last 10 messages
    message_history = deque(maxlen=10)
    cl.user_session.set("message_history", message_history)

    # Send the initial message with an introduction
    await cl.Message(
        author="Assistant",
        content=(
            "Hello! I'm an AI financial assistant. How may I help you?\n\n"
            "Here are some topics you can ask me about:\n"
            "- Budgeting and saving tips\n"
            "- Investment advice\n"
            "- Retirement planning\n"
            "- Understanding credit scores\n"
            "- Managing debt\n"
            "- Tax planning\n"
            "- Insurance options\n"
            "Feel free to ask any questions related to these topics!"
        )
    ).send()
async def set_sources(response, msg):
    elements = []
    label_list = []
    for count, sr in enumerate(response.source_nodes, start=1):
        elements.append(cl.Text(
            name="S" + str(count),
            content=f"{sr.node.text}",
            display="side",
            size="small",
        ))
        label_list.append("S" + str(count))
    msg.elements = elements
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    
    # Initialize message history as an empty list if not set
    message_history = cl.user_session.get("message_history", deque(maxlen=10))
    
    if not isinstance(message_history, list):
        message_history = []
        cl.user_session.set("message_history", message_history)

    # Generate a unique chat_id for this session
    chat_id = str(uuid4())
    payload = {
        "chat_id": chat_id,
        "message": message.content,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    
    # Save the message to Literal AI
    try:
        lai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[payload],
        )
        print("Message saved to Literal AI.")
    except Exception as e:
        print(f"Error saving message to Literal AI: {e}")
    
    msg = cl.Message(content="", author="Assistant")
    user_message = message.content
    
    # Process the user's query asynchronously
    res = agent.chat(message.content)
    
    # Check if 'res' has a 'response' attribute for the full response
    if hasattr(res, 'response'):
        # Send the full response in one go
        msg.content = res.response
        message_history.append({"author": "Human", "content": user_message})
        message_history.append({"author": "AI", "content": msg.content})
        message_history = list(message_history)[-4:]  # Keep the last 4 messages
        cl.user_session.set("message_history", message_history)
    else:
        # If res does not have a 'response' attribute, output a generic message
        msg.content = "I couldn't process your query. Please try again."
    await msg.send()
    if res.source_nodes:
        await set_sources(res, msg)

@cl.on_chat_resume
async def resume():
    try:
        # Ensure lai.chats.list() is awaited if it's an async function
        chat_history = await lai.chats.list() if asyncio.iscoroutinefunction(lai.chats.list) else lai.chats.list()
        
        if chat_history:
            for message in chat_history:
                # Append the message to local memory and send it to the chat interface
                await cl.Message(content=message['message']).send()
            print("Chat history loaded successfully!")
        else:
            print("No chat history found.")

    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        # await cl.Message(content="Error retrieving chat history. Please try again.").send() 
    # Send a message indicating the session has resumed
    await cl.Message(content="Welcome back! How can I assist you?").send()
    
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, str],
    default_user: cl.User
) -> Optional[cl.User]:
    """Handle Google OAuth callback."""
    print("OAuth callback received from provider:", provider_id)
    print("Token:", token)
    print("Raw user data:", raw_user_data)

    # Check if the provider is Google and process user information
    if provider_id == "google":
        user_email = raw_user_data.get("email")
        if user_email:
            return cl.User(identifier=user_email, metadata={"role": "user"})
    return None

