################################## 0.Import ##############################################
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage, AIMessage


import os
import torch
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

from langchain.document_loaders import JSONLoader
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch
import warnings
import openai
warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify

os.environ["OPENAI_API_KEY"] = ""
# GPT를 사용하기위한 API 키
openai.api_key = ""

################################## 1.모델 기본설정 맟 DB 호출 ##################################  

# flask 코드 추가
app = Flask(__name__)

# 모델 4bit로 다운받기
model_id = "vegegogi/woori_buddy_5.8b"

print('4bit')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available.")
    
print('모델과 토크나이저 다운') 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model.eval()
model.config.use_cache = True  # silence the warnings. Please re-enable for inference!

# 임베딩 함수 정의
embeddings = OpenAIEmbeddings()

# load from disk
DB = Chroma(persist_directory="./db/woori", embedding_function=embeddings)

######################################## 2.함수정의 ########################################

# 현재 어떤 태스크 진행하는지 확인하기 위한 문구 생성 함수 (평범한 업무)
def task_title_normal(message, total_length, fill_char='-'):
      adjusted_length = len(message) + sum(1 for char in message if ord(char) > 127)
      space_to_fill = total_length - adjusted_length
      left_padding = space_to_fill // 2
      right_padding = space_to_fill - left_padding
      formatted_message = fill_char * left_padding + message + fill_char * right_padding
      print(formatted_message,'\n')


# 현재 어떤 태스크 진행하는지 확인하기 위한 문구 생성 함수 (중요한 업무)
def task_title_important(message, total_length, fill_char='*'):
      adjusted_length = len(message) + sum(1 for char in message if ord(char) > 127)
      space_to_fill = total_length - adjusted_length
      left_padding = space_to_fill // 2
      right_padding = space_to_fill - left_padding
      formatted_message = fill_char * left_padding + message + fill_char * right_padding
      print(formatted_message, '\n')

# 생성형 AI를 통해 답변을 생성하기
def gen(question, context):
    inputs = tokenizer(
        f"### 질문:{question}? 맥락을 참조해서 답변하고, 알 수 없는 글자는 적절히 보정해줘. 만약 맥락에서 답변을 구할 수 없으면 답변하지마. \n\n{context} \n\n### 답변:",
        return_tensors='pt',
        return_token_type_ids=False
    )
    # 여기에서 input_ids를 cuda 디바이스로 옮깁니다.
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    # inputs = {k: v.to('cpu') for k, v in inputs.items()}

    gened = model.generate(
        **inputs,
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    return tokenizer.decode(gened[0])

# 생성형 AI의 답변을 전처리 하는 함수
def extract_first_answer(text):
    # "### 답변:" 뒤의 내용을 찾음
    start_index = text.find("### 답변:")
    if start_index == -1:
        return None
    start_index += len("### 답변:")

    # 답변 내용 중 "다."로 끝나는 첫번째 문장을 찾음
    end_index = text.find("다.", start_index) + len("다.")

    return text[start_index:end_index].strip()


# 모델 함수 호출
@app.route('/model_response', methods=['POST'])
def model_response():
     # 요청 데이터에서 "question" 항목을 가져옵니다.
    data = request.get_json()
    question = data.get("question")

    # DB에서 질문과 가장 유사한 질문을 가진 문서 찾기
    docs = DB.similarity_search(question)
    similar = docs[0].page_content

    # 찾은 문서를 전처리하기
    start_idx = similar.find("### 맥락:") + len("### 맥락:")
    end_idx = similar.find("### 답변:")

    context = "## 맥락:" + similar[start_idx:end_idx].strip() + " ### 답변:" + similar[end_idx+len("### 답변:"):].strip()

    # 질문과 문서를 기반으로 답변 생성
    answer = gen(question, context)
    
    # 전처리를 통해 최종 답변 생성
    result = extract_first_answer(answer)

    # 응답을 JSON 형식으로 반환합니다.
    return jsonify({'aaquestion': question, 'answer': result, 'best_context' : similar})


# gpt 호출
@app.route('/gpt_response', methods=['POST'])
def gpt_response():
      data = request.get_json()
      question = data.get("question")
      
      # GPT를 사용하기위한 API 키
      openai.api_key = ""
      
      prompt_template = """
      The user, a financial novice, has posed a question similar to {question}.
      The user finds answer is insufficient as a proper response to {question}.
      Hence, please craft a new answer, referencing {context} to supplement answer.
      If you don't know the answer, just say that you don't know, don't try to make up an answer.
      {context}

      Question: {question}
      Answer in Korean:
      """

      PROMPT = PromptTemplate(
          template=prompt_template,
          input_variables=["context", "question"])

      chain_type_kwargs = {"prompt": PROMPT}
      
      llm = ChatOpenAI(model_name="gpt-4", temperature=0)
      
      Retrieval_chain = RetrievalQA.from_chain_type(llm=llm,
                                         chain_type="stuff",
                                         retriever=DB.as_retriever(search_type="similarity",
                                                                          search_kwargs={"k":2}),
                                         return_source_documents=True,
                                         chain_type_kwargs=chain_type_kwargs,
                                         output_="result",
                                         )
      
      response_text = Retrieval_chain({"query": question})
      
      return jsonify({'response_text': response_text['result']})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
