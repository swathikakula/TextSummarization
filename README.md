# TextSummarization
 In recent world, data is essential for any organization growth. There’s  abundant data gathered all across. I believe summarization is key with data available in huge documents. 
 Extractive and abstractive text summarization are two techniques widely used. Extractive summarization, identifies top highly informative sentences. The key challenge aspect would be performing abstractive summarization ,where the model rephrases the data similar to the way humans do, by understanding the context of it.

Dataset:
Data will be download manually into local file system through links available in https://cs.nyu.edu/~kcho/DMQA/ .  The data is based on information in  ‘DailyMail  News’ articles. It contains approximately 10k documents (test data) and 197k documents (train data) which will be used as our dataset. The weblinks are available in two documents as:-
1.DailyMail_training_url
2.DailyMail_test_url
 The weblinks will be used to access the HTML pages which will be scraped to obtain the text and summaries.
The summary is given in HTML tags referring to <div> tag whose id=”js-article-text” which needs to be scraped.
Using supervised learning the textual data will be converted to short summaries. It will be a non-classification data and using a concept of abstractive summarization. Tackling this task is an important step towards natural language understanding. Abstractive Summarization generates a shorter version of a given sentence while attempting to preserve its meaning.


TFIDF
