import os
import boto3
import json
import sys
from utils import bedrock
from utils.TokenCounterHandler import TokenCounterHandler

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

module_path = ".."
sys.path.append(os.path.abspath(module_path))

bedrock_client = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
    runtime=True # Default. Needed for invoke_model() from the data plane
)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                               client=bedrock_client)

token_counter = TokenCounterHandler()

llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", 
              client=bedrock_client, 
              callbacks=[token_counter])


db = FAISS.load_local("final_faiss_claude_index", embeddings, allow_dangerous_deserialization=True)

query = ""

results_with_scores = db.similarity_search_with_score(query)

prompt_template = """

Human: Here is a set of context, contained in <context> tags:

<context>
{context}
</context>

You are a financial expert named DorNea, and you are very intelligent. You use financial terminology and are also very helpful, providing valuable and detailed information for me. ALWAYS ANSWER IN ENGLISH!! The context to provide an answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
    callbacks=[token_counter]
)

def askLLM(query):
    result = qa({"query": query})
    answer = result['result']
    source_documents = result['source_documents']
    sources=[]
    for source in source_documents:
            sources.append(source.metadata['source'])
    return answer, sources
