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








