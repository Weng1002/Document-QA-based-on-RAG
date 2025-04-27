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

其中的文字轉embedding模型，選擇 **intfloat/e5-large-v2**，原本是挑選**BAAI/bge-large-en-v1.5**，但前者在長文本的轉換任務(尤其英文)更為強大，但也需要更大的GPU支援。

1、 資料集前處理
```python
  def clean_text(text: str) -> str:
    # 1. 移除顯示用標記
    text = re.sub(r"(INLINEFORM\d+|DISPLAYFORM\d+|SECREF\d+|TABREF\d+|UID\d+|FIGREF\d+)", "", text)
    
    # 2. 移除 ::: 標記或章節分隔線
    text = re.sub(r"\s*:::+\s*", "\n", text)
    
    # 3. 移除參考文獻標題
    text = re.sub(r"(?i)\nreferences\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\(Table \d+\)|\(Figure \d+\)", "", text)  # 移除表格/圖表引用
    text = re.sub(r"\n\d+\s*$", "", text, flags=re.MULTILINE)  # 移除頁碼

    return text.strip()
```
*將一些亂碼給清洗掉，例如分段＂：：：＂、一些研討會期刊．．．*

2、 檢索⽅法
```python
  for demo_id, item in enumerate(tqdm(dataset, desc="QA 回答中...")):
      title = item["title"]
      full_text = clean_text(item["full_text"])
      question = item["question"]
  
      # 拆分文件段落
      documents = full_text.split("\n\n\n")
      docs = [Document(page_content=doc) for doc in documents]
  
      # 切 chunk
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=400,
          chunk_overlap=256,
          length_function=len,
          add_start_index=True,
      )
      docs_splits = text_splitter.split_documents(docs)
  
      vector_store = FAISS.from_documents(docs_splits, embedding_model)
  
      # === 動態 retrieval（逐步 k 增加）===
      retrieved_chunks = []
      max_k = RETRIEVE_TOP_K
  
      for k in range(1, max_k + 1):
          retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
          topk_docs = retriever.get_relevant_documents(question)
          retrieved_chunks = [doc.page_content for doc in topk_docs]
          reranked_sentences = rerank_sentences_by_similarity(question, retrieved_chunks, top_n=15)    
          context_text = "\n".join(reranked_sentences)
  
          # LLM 判斷是否足夠
          judge_result = confidence_chain.run({"context": context_text, "question": question}).strip().upper()
          if "YES" in judge_result:
              break  
```
*先根據""\n\n\n"去做大章節分割，然後再根據每章節使用LangChain中的"RecursiveCharacterTextSplitter"套件去切chunks，然後目前調整為最多為400個token然後其中的256個維overlap(0.64)，因為需要兩個段落必須要有overlap，不然會出現上下文不對襯現象。*

*原本是設計overlap達到0.72，但發現會造成反而每次切的chunk太少，且過多重複資訊，所以我發現overlap的比例也不能太高。*

*接著利用最常見的向量庫FAISS來製作，但這邊也有嘗試過使用ScaNN來實作，但後者的實作速度偏慢，且比較適合應用在超大量規模任務上，所以前者的優勢就比較明顯，可以適用我們這次的任務，且有更多的可控性可以去調整*

*然後這邊我自己多了個新的機制:動態 retrieval，主要是去動態地調整每一筆資料中所獲取的evidence，來提升ROUGE-L的分數，其中我建立Retrieval工具然後使用"similarity"，來找到top-k個最相似的chunks。*

```python
  sent_embed_model = SentenceTransformer("BAAI/bge-reranker-v2-m3")

def rerank_sentences_by_similarity(question, chunks, top_n=15, min_word_count=1):
    seen = set()
    sentences = []
    min_char_len = 10

    for chunk in chunks:
        for s in re.split(r'(?<=[.。!?])\s+', chunk):
            s = s.strip()
            
            if (len(s.split()) >= min_word_count and
                len(s) >= min_char_len and
                s not in seen):
                seen.add(s)
                sentences.append(s)

    # 計算語意相似度分數
    query_embedding = sent_embed_model.encode(question, convert_to_tensor=True)
    scored = [
        (s, util.pytorch_cos_sim(query_embedding, sent_embed_model.encode(s, convert_to_tensor=True)).item())
        for s in sentences
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_n]]
```

*將剛剛top-k = 25最相關的chunks，最進行一次reranking的動作，這邊使用BAAI/bge-reranker-v2-m3，來計算embedding的相似度，然後再挑出top-k=15的chunks，以降低llm撈到許多吳相關的chunks。*

```python
  # 信心判斷 prompt：請根據 evidence 判斷是否足以回答問題
  CONFIDENCE_PROMPT = PromptTemplate.from_template("""
  You are a QA validation model. Based on the following retrieved context and question, judge if the context provides enough information to confidently answer the question.
  
  Context:
  {context}
  
  Question:
  {question}
  
  Respond with only "YES" or "NO".
  """)
  confidence_chain = LLMChain(llm=llm, prompt=CONFIDENCE_PROMPT)
```

*然後再使用一個新機制，為confidence_chain，讓llm去判斷這些chunks是否足夠回答這個題目的question了，不夠就繼續retrival，來實現動態 retrieval，來讓每隔題目有不同數量的evidence。*
