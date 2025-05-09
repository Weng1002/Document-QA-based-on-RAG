# Document-QA-based-on-RAG
113-2 電機所 生成式AI HW3 Document QA based on RAG

## Author：國立陽明交通大學 資訊管理與財務金融學系財務金融所碩一 313707043 翁智宏

本次是生成式AI課程的第三次作業，是做 Document QA based on RAG，主要要利用開源的 LLM API，然後我們要將 "Public_datasets.json" 或是 "Private_datasets.json" 中的 "full_text" 去轉換成 embedding，然後設計出一個 RAG 系統，去使 LLM 可以從中找到回答 "Question" 的 "evidence"，來回答 "answer"。

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

這邊有做了三個版本的 RAG 實驗，然後到了最後一次才真正理解這次的作業詳細的步驟以及做法，所以接下來我簡單去解釋和區分我前面兩次實驗的哪裡不妥點。

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
*本次使用 "LangChain" 幫助快速搭建 RAG。*

```python
  DATASET_PATH = "datasets/public_dataset.json" #用來evaluate的
  OUTPUT_PATH = "sample_submission_public.json" #用來evaluate的
  
  # DATASET_PATH = "datasets/private_dataset.json" 
  # OUTPUT_PATH = "sample_submission_private.json"
  RETRIEVE_TOP_K = 30  #最多可以挑出top-k = 30個evidences
  
  embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
  # embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
  llm = ChatGroq(
      model="llama-3.1-8b-instant",
      api_key="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",  
      temperature=0.4,
      max_tokens=256
)
```
*我這次使用 Groq 的 API，然後挑選 llama-3 的模型來實作*

其中的文字轉 embedding 模型，選擇 **BAAI/bge-large-en-v1.5**，原本是挑選 **intfloat/e5-large-v2**，但前者在長文本的轉換任務(尤其英文)更為強大，但也需要更大的 GPU 支援。

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
*將一些亂碼給清洗掉，例如分段 ":::"、一些"研討會期刊"、Latex顯示用的標記*

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
          chunk_size=800,
          chunk_overlap=320,
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
          reranked_sentences = rerank_sentences_by_similarity(question, retrieved_chunks, top_n=20)    
          context_text = "\n".join(reranked_sentences)
  
          # LLM 判斷是否足夠
          judge_result = confidence_chain.run({"context": context_text, "question": question}).strip().upper()
          if "YES" in judge_result:
              break  
```
*先根據 "\n\n\n" 去做大章節分割，然後再根據每章節使用 LangChain 中的 "RecursiveCharacterTextSplitter" 套件去切 chunks，然後目前調整為最多為 800個token 然後其中的 320個overlap(0.4)，因為需要兩個段落必須要有 overlap，不然會出現上下文不對襯現象。*

*原本是設計 overlap 達到 0.72，但發現會造成反而每次切的 chunk 太少，且過多重複資訊，所以我發現 overlap 的比例也不能太高。*

*接著利用最常見的向量庫 FAISS 來製作，但這邊也有嘗試過使用ScaNN來實作，但後者的實作速度偏慢，且比較適合應用在超大量規模任務上，所以前者的優勢就比較明顯，可以適用我們這次的任務，且有更多的可控性可以去調整。*

*然後這邊我自己多了個新的機制:動態 retrieval，主要是去動態地調整每一筆資料中所獲取的 evidence，來提升 ROUGE-L 的分數，其中我建立 Retrieval 工具然後使用 "similarity"，來找到 top-k 個最相似的 chunks。*

```python
sent_embed_model = SentenceTransformer("BAAI/bge-reranker-large")

def rerank_sentences_by_similarity(question, chunks, top_n=20, min_word_count=1):
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

*將剛剛 top-k = 30 最相關的 chunks，最進行一次 reranking 的動作，這邊使用 BAAI/bge-reranker-large，來計算 embedding 的相似度，然後再挑出 top-k=20 的 chunks，以降低 llm 撈到許多吳相關的 chunks。*

```python
  # 信心判斷 prompt：請根據 evidence 判斷是否足以回答問題
  CONFIDENCE_PROMPT = PromptTemplate.from_template("""
  You are a QA validation model. Given the retrieved context and the question, determine if the context contains enough information to answer the question confidently and correctly.
  
  Criteria:
  - If the context covers more than 70% of the necessary information to answer, respond "YES".
  - If the information is clearly incomplete or unrelated, respond "NO".
  
  Context:
  {context}
  
  Question:
  {question}
  
  Respond with only "YES" or "NO".
  """)
  
  confidence_chain = LLMChain(llm=llm, prompt=CONFIDENCE_PROMPT)
```

*然後再使用一個新機制，為 confidence_chain，讓 llm 去判斷這些 chunks 是否足夠回答這個題目的 question 了，不夠就繼續 retrieval，來實現動態 retrieval，來讓每隔題目有不同數量的 evidence。*

3、 Prompt 技巧
```python
  CHAT_TEMPLATE_RAG = (
    """human: You are an academic QA assistant. Use the context to answer precisely.
    Please think about the question step by step, and then answer a ***concise***, precise answer based on the context and evidence.
    Please try to find the right keywords to answer the question based on the evidence or context you find.
    If the answer is a name, number, or keyword, extract it directly.
    Or write a short, complete answer (1–3 sentences) based only on the context.
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

*這邊使用到的 prompt engineering，先讓 llm 扮演角色(role)，然後使用類似 CoT 的方式，請模型要仔細思考剛剛得到的 evience，透過這些證據去回答問題，然後要求不同只根據自身 llm 的能力來回答答案，要有依據的回答。然後最後要求輸出的格式，要類似 SQuAD or CoQA 以及答案是關鍵字或是數字類型，直接回答就好，不要有過多的贅述。*

原本我使用的 Prompt：
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

*這份 prompt，就顯得太冗贅且我有實驗加入 few-shot，會讓我的 llm 回答的方式都跟給的範例一樣，從而限縮回答的多樣性。*

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

*其中這邊我也有設計一個後處理機制，是讓 "evidence" 的句子不是只有一個單詞，去調整 len，來確保至少 retrieval 的句子相較來說比較可信。*

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

*這邊有去調整助教原本給的評估機制，去調整成適應多筆資料的方式。去平均每筆的 ROUHE-L，得到一個總平均 ROUHE-L。*


### 輸出結果
> public_submisson.json (只挑前3題)：

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
    }
```

然後 public_submisson 的正解:
```
  "answer": [
        "During training, the model is trained alternately with one mini-batch of labeled data and INLINEFORM0 mini-batches of unlabeled data."
      ],
      "evidence": [
        "CVT BIBREF20 is a semi-supervised learning technique whose goal is to improve the model representation using a combination of labeled and unlabeled data. During training, the model is trained alternately with one mini-batch of labeled data and INLINEFORM0 mini-batches of unlabeled data.",
        "Labeled data are input into the model to calculate the standard supervised loss for each mini-batch and the model weights are updated regularly. Meanwhile, each mini-batch of unlabeled data is selected randomly from the pool of all unlabeled data; the model computes the loss for CVT from the mini-batch of unlabeled data. This CVT loss is used to train auxiliary prediction modules, which see restricted views of the input, to match the output of the primary prediction module, which is the full model that sees all the input. Meanwhile, the auxiliary prediction modules share the same intermediate representation with the primary prediction module. Hence, the intermediate representation of the model is improved through this process.",
        "As discussed in Section SECREF3 , CVT requires primary and auxiliary prediction modules for training with unlabeled data to improve the representation. Thus, we construct both types of prediction modules for our model. The flow of unlabeled data, which is processed to obtain a prediction by each module, is shown in Fig. . The output of each prediction module is transformed into the probability distribution of each class by the softmax function and then used to calculate INLINEFORM0 , as shown in cvtloss. DISPLAYFORM0"
      ]
    },
  "answer": [
        "English "
      ],
      "evidence": [
        "We trained our error generation models on the public FCE training set BIBREF16 and used them to generate additional artificial training data. Grammatically correct text is needed as the starting point for inserting artificial errors, and we used two different sources: 1) the corrected version of the same FCE training set on which the system is trained (450K tokens), and 2) example sentences extracted from the English Vocabulary Profile (270K tokens).. While there are other text corpora that could be used (e.g., Wikipedia and news articles), our development experiments showed that keeping the writing style and vocabulary close to the target domain gives better results compared to simply including more data."
      ]
    },
  "answer": [
        "(+1 or -1), words of opposite polarities (e.g. “happy\" and “unhappy\") get far away from each other"
      ],
      "evidence": [
        "Only the information concerned with the dictionary definitions are used there, discarding the polarity scores. However, when we utilise the supervised score (+1 or -1), words of opposite polarities (e.g. “happy\" and “unhappy\") get far away from each other as they are translated across coordinate regions."
      ]
    }
```

> ROUGE-L：0.2541

> private_submisson.json (只挑前3題)：
```
    {
      "title": "How Document Pre-processing affects Keyphrase Extraction Performance",
      "answer": "The SemEval-2010 benchmark dataset BIBREF0 is composed of 244 scientific articles. Therefore, there are 244 articles in the dataset.",
      "evidence": [
        "The SemEval-2010 benchmark dataset BIBREF0 is composed of 244 scientific articles collected from the ACM Digital Library (conference and workshop papers). The input papers ranged from 6 to 8 pages and were converted from PDF format to plain text using an off-the-shelf tool. The only preprocessing applied is a systematic dehyphenation at line breaks and removal of author-assigned keyphrases. Scientific articles were selected from four different research areas as defined in the ACM classification, and were equally distributed into training (144 articles) and test (100 articles) sets. Gold standard keyphrases are composed of both author-assigned keyphrases collected from the original PDF files and reader-assigned keyphrases provided by student annotators. Long documents such as those in the"
      ]
    },
    {
      "title": "Comparative Studies of Detecting Abusive Language on Twitter",
      "answer": "The authors provide two examples to illustrate the difficulties presented by the context-dependent nature of online aggression. The first example involves two tweets: (1) \"I hate when I'm sitting in front of the bus and somebody with a wheelchair get on\" and (2) \"I hate it when I'm trying to board a bus and there's already an as**ole on it.\" The second tweet is labeled as abusive due to the use of vulgar language, but understanding its context tweet (1) clarifies the user's intention. Another example is mentioned with context tweet (3), although the specific content of (3) is not provided. These examples demonstrate how considering \"context tweets\" (i.e., the tweet being replied to or quoted) can provide significant contextual information to better understand the intention behind a potentially abusive tweet.",
      "evidence": [
        "The context-dependent nature of online aggression makes annotating large collections of data extremely difficult. Previously studied datasets in abusive language detection have been insufficient in size to efficiently train deep learning models. Recently, Hate and Abusive Speech on Twitter, a dataset much greater in size and reliability, has been released. However, this dataset has not been comprehensively studied to its potential. In this paper, we conduct the first comparative study of various learning models on Hate and Abusive Speech on Twitter, and discuss the possibility of using additional features and context data for improvements. Experimental results show that bidirectional GRU networks trained on word-level features, with Latent Topic Clustering modules, is the most accurate",
        "While manually analyzing the raw dataset, we noticed that looking at the tweet one has replied to or has quoted, provides significant contextual information. We call these, “context tweets\". As humans can better understand a tweet with the reference of its context, our assumption is that computers also benefit from taking context tweets into account in detecting abusive language. As shown in the examples below, (2) is labeled abusive due to the use of vulgar language. However, the intention of the user can be better understood with its context tweet (1). (1) I hate when I'm sitting in front of the bus and somebody with a wheelchair get on.   (2) I hate it when I'm trying to board a bus and there's already an as**ole on it.  Similarly, context tweet (3) is important in understanding the"
      ]
    },
    {
      "title": "Prepositional Attachment Disambiguation Using Bilingual Parsing and Alignments",
      "answer": "Enforcing agreement between parse trees across different languages works by ensuring that the dependency edges between aligned words in the two languages correspond to each other. Specifically, if two words in the English sentence (i and i') are aligned to words in the Hindi sentence (j and j'), a dependency edge between i and i' in the English parse tree should correspond to an edge between j and j' in the Hindi parse tree, and vice versa. This allows the Dependency Driven (DD) algorithm to synchronize the parse trees of the two languages based on word alignments.",
      "evidence": [
        "sentence pair, with its word alignments given. We first obtain the predicted parse trees for the English and Hindi sentences from the respective trained parser models as an initialsiation step. The DD algorithm then tries to enforce agreement between the two parse trees subject to the given alignments. Let us take a closer look at what we mean by agreement between the two parse trees. Essentially, if we have two words in the English sentence denoted by i and i', aligned to words j and j' in the parallel Hindi sentence respectively, we can expect a dependency edge between i and i' in the English parse tree to correspond to an edge between j and j' in the Hindi parse tree, and vice versa. Also, in order to accommodate structural diversity in languages BIBREF1 , we can expect an edge in the"
      ]
    }
```

*可以看出我的 evidence，每個題目都會是動態調整，然後 answer，會根據我剛剛的 prompt 設計，當有些題目的輸出，就直接是答案，或是設計多一些解釋性回答，因為 llama-4，比較擅長回答一段話，也可以讓我這個回答更能被判斷成 correctness，來提升 correctness。*

### 輸出圖表呈現

| - | 0415-benchmark |  0421-benchmark |  0426-benchmark |  0428-benchmark |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| ROUGE-L | 0.0883 | 0.1189 | 0.2159 | 0.2310 |
| Correctness | 0.02 | 0 | 0.34 | 0.38 |

> 可以看到前兩次真的都不清楚這任務的方向，所以表現都很差，後面兩次就知道方向，然後就開始進入高效微調的階段了

## 發現的觀點與後話

### 輸出分析
  從我的private輸出的結果可以發現，我的 "answer"，都會再去額外多一些解釋性的句子，但因為 ROUGE-L 主要是看句子的 Overlap，所以我這邊就盡量不要讓輸出只有一個單詞，能有一些解釋性句子更好，我的目的是希望能讓指標更高一點。

  然後我也發現我有一些 "evidence" 都還是會 retrieval 到一些無關的句子，原本自己認為這部分可以透過 re-retrieval 去再次針對第一次所找到的 "evidence" 再去處理，但可能是我設計不良，反而實作這機制會讓我的 ROUGE-L 降低，所以最後還是沒有實現這機制，但取而代之就是，去提升 Chunks 的大小，來讓 LLM 能更讀得懂段落，來找到更相近的 "evidence" ，且也搭配更強大的 Embedding 模型，來去彌補這部分的弱勢，還有搭配動態挑選 top-k 個句子，發現使用這個機制，會讓我的指標表現提升5-6%，影響很大！
  
### 與其他版本的差異
  然後我有提到我有做三個版本，最後一個版本才有正常的成功，是因為我第一份發現我完全搞錯這次的作業方向，我把 Public 當成是要用在 Private 中找相關答案的資料庫，導致我看我前期的輸出都很怪，感覺都沒有跟題目很相近，於是第一個版本的 code，用了許多不同的機制像是：
```
    # 按論文結構分割
    def split_by_sections(text):
      sections = re.split(
          r'<SECTION>|(?=Abstract\n|Introduction\n|Related Work\n|Background\n|Data\n|Approach\n|Methodology\n|Evaluation\n|Experiments\n|Conclusion\n|Acknowledgements\n|(?:\w+\s*:::\s*.+\n))',
          text
      )
      return [s.strip() for s in sections if s.strip()]
```

我去根據論文中的更細的章節切割，但發現這樣會導致往往答案都在 "abstract" 或是 "introduction" 的後半部分，都被遺忘掉。

還設計了，二次驗證答案的機制：
```
  def validate_answer(data):
      issues = []
      for item in data:
          full_text = clean_text(item['full_text'])  # 清理 full_text
          answers = item['answer']
          evidence = item['evidence']
          
          # 檢查答案
          for ans in answers:
              cleaned_ans = clean_text(ans)
              if cleaned_ans not in full_text and cleaned_ans.lower() not in full_text.lower():
                  issues.append(f"Answer '{ans}' not found in full_text for title: {item['title']}")
          
          # 檢查證據
          for ev in evidence:
              cleaned_ev = clean_text(ev)
              ev_words = set(cleaned_ev.lower().split())
              full_text_words = set(full_text.lower().split())
              overlap = len(ev_words & full_text_words) / len(ev_words) if ev_words else 0
              if overlap < 0.8:
                  issues.append(f"Evidence '{ev}' not found or insufficient overlap ({overlap:.2f}) in full_text for title: {item['title']}")
      
      return issues
```

拿輸出後的答案，再去給 llm 找答案有無出現在 "Question"、"evidence" 中，但因為這邊我搞錯任務方向，導致這機制的表現也很混亂，最後我也就捨棄這方法。

最後還有一種
```
  def is_how_many_question(q):
      return q.lower().strip().startswith("how many")
  
  if is_how_many_question(question):
          # 處理 'How many' 類問題，返回相關的數字答案
          answer_text = f"There are {len(public_data)} articles in the dataset."
          qa_result = qa_chain.invoke({"query": question})
          evidence_list = [doc.page_content for doc in qa_result.get("source_documents", [])]
      else:
          qa_result = qa_chain.invoke({"query": question})
          cot_output = qa_result.get("answer") or qa_result.get("result") or "I don't know"
          answer_text = extract_final_answer(cot_output)
          evidence_list = [doc.page_content for doc in qa_result.get("source_documents", [])]
```

去針對 "計數" 的題型處理，因為我原本以為這份作業的輸出要像 public_dataset.json 一樣，答案用 list 來包裝，所以我就強制去調整判斷，當遇到計數題型，直接輸出看到的數字就是這題的答案，但這機制後面實驗看來，也是很荒謬，因為第二、第三個實驗得知，只需要將 llm 的 prompt 設計好，以及 retrieval 機制整好就可以避免上述的問題了。

然後第二個實驗，我就知道搞錯方向，於是我參考助教給予的那份 rag-v2.ipynb 去修改，然後也將輸出丟進去 evaluation 中測試，但發現還是不太好，最後測試了幾次，最終在第三個版本中終於發現是因為，我原先是將所有的 full_text 都做成 FAISS 向量庫，導致每個題目在找對應的 evidence 都要從很龐大的向量庫中去取得對應的關鍵字，造成準確度以及冗言贅字增加，於是我的最終版就將這部分改成，每筆資料都獨立轉成 FAISS 向量庫，這樣做雖然後增加運算時間但也讓我的表現提升了很多%！

### 省思
最後我透過這次的任務更了解到了，包含前面兩個任務(Prompt Engineering、Paper-Abstract-Generation)的綜合體 RAG 機制，這次的任次讓我感覺我對於一套很基礎的 RAG 的整套流程該如何設計、要如何做好資料前處理、讓 LLM 可以更聽話等等的操作，都在這次的實驗中體驗到了！所以在最後的成績兩者指標都有超過 Strong 感到很滿意！唯一的可惜是，我挑的模型參數量比較大，達到 8B，在這部分可能還有更多的調整空間去探討！

