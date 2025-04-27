# Document-QA-based-on-RAG
113-2 電機所 生成式AI HW3 Document QA based on RAG

## Author：國立陽明交通大學 資訊管理與財務金融學系財務金融所碩一 313707043 翁智宏

本次是生成式AI課程的第三次作業，是做 Document QA based on RAG，主要要利用開源的LLM API，然後我們要將"Public_datasets.json"或是"Private_datasets.json"中的"full_text"去轉換成embedding，然後設計出一個RAG系統，去使LLM可以從中找到回答"Question"的"evidence"，來回答"answer"。

## Dataset & files

The public dataset contains 100 papers with questions, as does the private dataset.
The only difference between private and public dataset is that there is no “answer” and “evidence” in private dataset. Since these are what we are going to predict (and retrieve)!
- public_dataset.json: For the development purpose. You can use this data to develop an efficient RAG system.
- private_dataset.json: For the ranking purpose.
- sample_submission.json: A sample submission file in the correct format, demonstrating expected output

然後透過兩種方式的評估指標：

1、 Answer correctness: Evaluated by LLM 
- You get either 0 or 1

2、 ROUGE-L: Evidence f-measure score
- Calculate ROUGE-L for all retrieved documents against the ground truth evidence and take the maximum value.
- If there are multiple ground truth evidence, calculate the average of each generated maximum ROUGE-L value.
- To be more specific: We will take n ground truth evidence and calculate ROUGE-L with the retrieved K documents, each n will get K scores, we take the maximum value from each n, this way we have K maximum values, then we take the average of them, which is the final score

## 實作

這邊有做了三個版本的RAG實驗，然後到了最後一次才真正理解這次的作業詳細的步驟以及做法，所以接下來我簡單去解釋和區分我前面兩次實驗的哪裡不妥點。

我將這次RAG作業，分成：

- Step 1：資料清洗＋段落切分＋向量化
- Step 2：建構向量檢索器（Retriever）
- Step 3：設計 Prompt 模板 & RAG 回答模組
- Step 4：處理 QA 任務
- Step 5：Evaluation

### ⽅法
0、 載入必要套件
```bash
  import json
  import re
  import torch
  from tqdm import tqdm
  from langchain.docstore.document import Document
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from langchain.embeddings import HuggingFaceEmbeddings
  from langchain.vectorstores import FAISS
  from langchain.chains.retrieval import create_retrieval_chain
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain.chains.llm import LLMChain
  from langchain_core.prompts import PromptTemplate
  from langchain_groq import ChatGroq
  from langchain.docstore.document import Document
  
  from sentence_transformers import SentenceTransformer, util
```
*本次使用"LangChain"幫助快速搭建RAG。*

```python
  DATASET_PATH = "datasets/public_dataset.json" #用來evaluate的
  OUTPUT_PATH = "sample_submission_public.json" #用來evaluate的
  
  # DATASET_PATH = "datasets/private_dataset.json" 
  # OUTPUT_PATH = "sample_submission_private.json"
  RETRIEVE_TOP_K = 25  #最多可以挑出top-k = 25個evidences
  
  embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")  
  llm = ChatGroq(
      model="meta-llama/llama-4-scout-17b-16e-instruct", 
      api_key="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",  
      temperature=0.4,
      max_tokens=256
)
```
*我這次使用Groq的API，然後挑選llama-4的模型來實作*
其中的文字轉embedding模型，選擇"intfloat/e5-large-v2"，原本是挑選"BAAI/bge-large-en-v1.5"，但前者在長文本的轉換任務(尤其英文)更為強大，但也需要更大的GPU支援。

1、 資料集前處理






