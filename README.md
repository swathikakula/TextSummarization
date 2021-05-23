# TextSummarization
 In recent world, data is essential for any organization growth. There’s  abundant data gathered all across. I believe summarization is key with data available in huge documents. 
 Extractive and abstractive text summarization are two techniques widely used. Extractive summarization, identifies top highly informative sentences. The key challenge aspect would be performing abstractive summarization ,where the model rephrases the data similar to the way humans do, by understanding the context of it.

Dataset:
DailyMail News articles are made avilable by  and the dataset can be accessed using link https://cs.nyu.edu/~kcho/DMQA/.
Once the files are downloaded locally,it contains directory containing weburl. It include approximately 10k documents (test data) and 197k documents (train data). The weblinks can be scrapped using data scraping techniques to obtain the text and summaries. 

In addition, theres another directory providing extracted summary and text associated to each story/article. Using python regex, summaries and text can be further seggreated. 
 
Model:
Using ML techniques such as TFIDF (term frequency Inverse document frequency), the data can be 
