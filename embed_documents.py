import boto3
import json
import os
import sys
import numpy as np
from utils.TokenCounterHandler import TokenCounterHandler
from pypdf import PdfReader, PdfWriter
import glob

# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

bedrock_client = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
    runtime=True # Default. Needed for invoke_model() from the data plane
)

token_counter = TokenCounterHandler()

# - create the Anthropic Model
llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", 
              client=bedrock_client, 
              callbacks=[token_counter])


# - create the Titan Embeddings Model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_client)

data_root = "data/"

local_pdfs = glob.glob(data_root + '*.pdf')

for local_pdf in local_pdfs:
    pdf_reader = PdfReader(local_pdf)
    pdf_writer = PdfWriter()
    for pagenum in range(len(pdf_reader.pages)-3):
        page = pdf_reader.pages[pagenum]
        pdf_writer.add_page(page)

    with open(local_pdf, 'wb') as new_file:
        new_file.seek(0)
        pdf_writer.write(new_file)
        new_file.truncate()
        

documents = []
root = 'data/'
for idx, file in enumerate(local_pdfs):
    loader = PyPDFLoader( file)
    document = loader.load()
    '''for document_fragment in document:
        document_fragment.metadata = metadata[idx]'''
        
    print(f'{len(document)} {document}\n')
    documents += document

# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
)

docs = text_splitter.split_documents(documents)
