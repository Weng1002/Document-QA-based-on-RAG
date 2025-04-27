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

3、 Prompt 技巧
```python
    CHAT_TEMPLATE_RAG = (
      """human: You are an academic QA assistant. Use the context to answer precisely.
  Please think about the question step by step, and then answer a ***concise***, precise answer based on the context and evidence.
  Please try to find the right keywords to answer the question based on the evidence or context you find.
  If the answer is a name, number, or keyword, extract it directly.
  Avoid vague or overly broad answers. Answer in a concise phrase.
  Format your answer similarly to human-written academic answers from datasets like SQuAD or CoQA.
  
  Context:  
  {context}
  
  Question:
  {input}
  
  assistant:"""
  )
  
  retrieval_qa_prompt = PromptTemplate.from_template(template=CHAT_TEMPLATE_RAG)
```

*這邊使用到的prompt engineering，先讓llm扮演角色(role)，然後使用類似CoT的方式，請模型要仔細思考剛剛得到的evience，透過這些證據去回答問題，然後要求不同只根據自身llm的能力來回答答案，要有依據的回答。然後最後要求輸出的格式，要類似 SQuAD or CoQA以及答案是關鍵字或是數字類型，直接回答就好，不要有過多的贅述。*

原本我使用的Prompt：
```python
  PROMPT_TEMPLATE = """
  You are a research assistant. Answer the question strictly based on the provided context and any clues or evidence.
  Carefully analyze the context and extract only the most relevant information.
  Provide a concise, direct answer to the question in a single sentence. 
  If the context provides no useful information, try your best to infer an answer from what you have.
  Only respond with "I don't know" if absolutely no relevant clues can be found in the context.
  Avoid explanations, introductory phrases, or unnecessary details. No extra steps or reasoning, just the final answer.
  Do NOT copy full sentences verbatim from the context.
  If the answer is too cryptic and unclear, just help me answer based on known clues to minimize the chances of answering “I don't know”.
  
  Context:
  {context}
  
  Question:
  {question}
  
  Answer (no explanation):
"""
```

*這份prompt，就顯得太冗贅且我有實驗加入few-shot，會讓我的llm回答的方式都跟給的範例一樣，從而限縮回答的多樣性。*

4、 QA回答
```python
  # === 啟用 RAG QA chain ===
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_prompt)
    rag_qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )
    response = rag_qa_chain.invoke({"input": question})
    predicted_answer = response["answer"].strip()
    
    # 只保留不是單字的句子（整段當作一句處理）
    predicted_evidence = []
    for doc in response["context"]:
        s = doc.page_content.strip()
        if len(s.split()) >= 2:
            predicted_evidence.append(s)
    
    sample_submission.append({
        "title": title,
        "answer": predicted_answer,
        "evidence": predicted_evidence
    })
```

*當上述的前處理跟模型都搭建好後，就開始進入推理的階段。*

*其中這邊我也有設計一個後處理機制，是讓"evidence"的句子不是只有一個單詞，去調整len，來確保至少retrival的句子相較來說比較可信。*

5、 Evaluate
```python
    from rouge_score import rouge_scorer
  
  scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
  
  def compute_evidence_rouge(gt_evidence, predicted_chunks):
      """計算一筆資料中 retrieved evidence 和 ground truth 之間的 ROUGE-L 平均 f1 分數"""
      if not predicted_chunks or not gt_evidence:
          return 0.0
  
      f_scores = []
      for pred in predicted_chunks:
          scores = scorer.score_multi(
              targets=gt_evidence,
              prediction=pred,
          )
          f_scores.append(scores["rougeL"].fmeasure)
      
      return sum(f_scores) / len(f_scores)
  
  # 在迴圈結束後，整體計算所有筆數的 evidence score
  total_score = 0.0
  valid_count = 0
  
  for i, item in enumerate(dataset):
      gt_evidence = item["evidence"]  # 標準答案中的 evidence sentences
      pred_evidence = sample_submission[i]["evidence"]  # 模型取出的句子
  
      if pred_evidence and gt_evidence:
          score = compute_evidence_rouge(gt_evidence, pred_evidence)
          total_score += score
          valid_count += 1
  
  average_score = total_score / valid_count if valid_count > 0 else 0.0
  print(f"[Total]: {valid_count} samples with valid evidence")
  print(f"[Average ROUGE-L Evidence F1]: {average_score:.4f}")
```

*這邊有去調整助教原本給的評估機制，去調整成適應多筆資料的方式。去平均每筆的ROUHE-L，得到一個總平均ROUHE-L。*


### 輸出結果
>　public_submisson.json：

```
{
    "title": "Semi-supervised Thai Sentence Segmentation Using Local and Distant Word Representations", 
    "answer": "They utilize unlabeled data to improve model representations by constructing prediction modules and processing the flow of unlabeled data through each module.", 
    "evidence": [
      "for training with unlabeled data to improve the representation. Thus, we construct both types of prediction modules for our model. The flow of unlabeled data, which is processed to obtain a prediction by each module, is shown in Fig. . The output of each prediction module is transformed into the probability distribution of each class by the softmax function and then used to calculate  , as shown"
    ]
  },
  {
      "title": "Artificial Error Generation with Machine Translation and Syntactic Patterns",
      "answer": "English.",
      "evidence": [
        "task, where grammatically correct text is translated to contain errors. In addition, we explore a system for extracting textual patterns from an annotated corpus, which can then be used to insert errors into grammatically correct sentences. Our experiments show that the inclusion of artificially generated errors significantly improves error detection accuracy on both FCE and CoNLL 2014 datasets.",
        "showed that keeping the writing style and vocabulary close to the target domain gives better results compared to simply including more data. We evaluated our detection models on three benchmarks: the FCE test data (41K tokens) and the two alternative annotations of the CoNLL 2014 Shared Task dataset (30K tokens) BIBREF3 . Each artificial error generation system was used to generate 3 different",
        "models. Figure  shows the  score on the development set, as the training data is increased by using more translations from the n-best list of the SMT system. These results reveal that allowing the model to see multiple alternative versions of the same file gives a distinct improvement – showing the model both correct and incorrect variations of the same sentences likely assists in learning a",
        "Profile (270K tokens).. While there are other text corpora that could be used (e.g., Wikipedia and news articles), our development experiments showed that keeping the writing style and vocabulary close to the target domain gives better results compared to simply including more data. We evaluated our detection models on three benchmarks: the FCE test data (41K tokens) and the two alternative",
        "amount of available annotated data is very limited. Rei2016 showed that while some error detection algorithms perform better than others, it is additional training data that has the biggest impact on improving performance. Being able to generate realistic artificial data would allow for any grammatically correct text to be transformed into annotated examples containing writing errors, producing",
        "translations. Therefore, we also investigated the combination of multiple error-generated versions of the input files when training error detection models. Figure  shows the  score on the development set, as the training data is increased by using more translations from the n-best list of the SMT system. These results reveal that allowing the model to see multiple alternative versions of the same",
        "This paper investigated two AEG methods, in order to create additional training data for error detection. First, we explored a method using textual patterns learned from an annotated corpus, which are used for inserting errors into correct input text. In addition, we proposed formulating error generation as an MT framework, learning to translate from grammatically correct to incorrect sentences.",
        "patterns learned from an annotated corpus, which are used for inserting errors into correct input text. In addition, we proposed formulating error generation as an MT framework, learning to translate from grammatically correct to incorrect sentences. The addition of artificial data to the training process was evaluated on three error detection annotations, using the FCE and CoNLL 2014 datasets.",
        "factor of dataset size and only focusing on the model architectures. The error generation methods can generate alternative versions of the same input text – the pattern-based method randomly samples the error locations, and the SMT system can provide an n-best list of alternative translations. Therefore, we also investigated the combination of multiple error-generated versions of the input files",
        "but also restricting the approach to only five error types. There has been very limited research on generating artificial data for all types, which is important for general-purpose error detection systems. For example, the error types investigated by Felice2014a cover only 35.74% of all errors present in the CoNLL 2014 training dataset, providing no additional information for the majority of",
        "Shortage of available training data is holding back progress in the area of automated error detection. This paper investigates two alternative methods for artificially generating writing errors, in order to create additional resources. We propose treating error generation as a machine translation task, where grammatically correct text is translated to contain errors. In addition, we explore a",
        "Machine Translation",
        "statistical significance and found that the improvement for each of the systems using artificial data was significant over using only manual annotation. In addition, the final combination system is also significantly better compared to the Felice2014a system, on all three datasets. While Rei2016 also report separate experiments that achieve even higher performance, these models were trained on a",
        "version of the same FCE training set on which the system is trained (450K tokens), and 2) example sentences extracted from the English Vocabulary Profile (270K tokens).. While there are other text corpora that could be used (e.g., Wikipedia and news articles), our development experiments showed that keeping the writing style and vocabulary close to the target domain gives better results compared",
        "generation as an MT framework, learning to translate from grammatically correct to incorrect sentences. The addition of artificial data to the training process was evaluated on three error detection annotations, using the FCE and CoNLL 2014 datasets. Making use of artificial data provided improvements for all data generation methods. By relaxing the type restrictions and generating all types of",
        "or noun number errors BIBREF2 . Felice2014a investigated the use of linguistic information when generating artificial data for error correction, but also restricting the approach to only five error types. There has been very limited research on generating artificial data for all types, which is important for general-purpose error detection systems. For example, the error types investigated by",
        "alignments of similar sentences often result in the same binary labeling. Future work could explore more advanced alignment methods, such as proposed by felice-bryant-briscoe. In Section  , this automatically labeled data is then used for training error detection models.",
        "input text – the pattern-based method randomly samples the error locations, and the SMT system can provide an n-best list of alternative translations. Therefore, we also investigated the combination of multiple error-generated versions of the input files when training error detection models. Figure  shows the  score on the development set, as the training data is increased by using more",
        "learners as the target. Pialign BIBREF6 is used to create a phrase translation table directly from model probabilities. In addition to default features, we add character-level Levenshtein distance to each mapping in the phrase table, as proposed by Felice:2014-CoNLL. Decoding is performed using Moses BIBREF7 and the language model used during decoding is built from the original erroneous",
        "of the SMT system. These results reveal that allowing the model to see multiple alternative versions of the same file gives a distinct improvement – showing the model both correct and incorrect variations of the same sentences likely assists in learning a discriminative model.",
        "work on artificial error generation (AEG) has focused on specific error types, such as prepositions and determiners BIBREF0 , BIBREF1 , or noun number errors BIBREF2 . Felice2014a investigated the use of linguistic information when generating artificial data for error correction, but also restricting the approach to only five error types. There has been very limited research on generating",
        "is performed using Moses BIBREF7 and the language model used during decoding is built from the original erroneous sentences in the learner corpus. The IRSTLM Toolkit BIBREF8 is used for building a 5-gram language model with modified Kneser-Ney smoothing BIBREF9 ."
      ]
    },
    {
      "title": "Generating Word and Document Embeddings for Sentiment Analysis",
      "answer": "The supervised scores of the words are calculated by scanning through all of its contexts and extracting the minimum, maximum, and average polarity scores of words occurring in the same contexts as the target word, in addition to the target word's self polarity score.",
      "evidence": [
        "Our last component is a simple metric that uses four supervised scores for each word in the corpus. We extract these scores as follows. For a target word in the corpus, we scan through all of its contexts. In addition to the target word's polarity score (the self score), out of all the polarity scores of words occurring in the same contexts as the target word, minimum, maximum, and average scores"
      ]
    },
```
