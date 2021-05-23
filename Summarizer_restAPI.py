import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import torch
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer
    )

import re
import os

import pandas as pd
import numpy as np
import string
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from summarizer import Summarizer
import re
import os
import nltk 
import pandas as pd
import numpy as np
import string
from nltk.tokenize import sent_tokenize

app =Flask(__name__) 
api = Api(app) 
#custom_var = reqparse.RequestParser() 
#custom_var.add_argument('text', type =str, help ='Enter text') 

class Textsummarization():
    @app.route('/Textsummary/abstractive', methods=['GET'])
    def t5finetuned():
        # use parser and find the user's query
        #args = custom_var.parse_args() 
        #User_input = args ['query'] 
        model = T5ForConditionalGeneration.from_pretrained('/Users/swathik/Desktop/t5')
        tokenizer= T5Tokenizer.from_pretrained('t5-base')
        model.eval()
        data= "summarize: "+request.form['text']
        input_ids = tokenizer.encode(data,return_tensors='pt')
        output = model.generate(input_ids,  min_length=100, max_length=150)

        #input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

        #generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)


        #preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in output][0]
        #summary = tokenizer.decode(output[1])
        
        return jsonify ({"T5 fine tuned summary": summary})

    @app.route('/Textsummary/abstractive_pretrained', methods=['GET'])
    def t5pretrained():
        # use parser and find the user's query
        #args = custom_var.parse_args() 
        #User_input = args ['query'] 
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer= T5Tokenizer.from_pretrained('t5-base')
        model.eval()
        data= "summarize: "+request.form['text']
        input_ids = tokenizer.encode(data,return_tensors='pt')
        output = model.generate(input_ids,  num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=100,
                                    max_length=150,
                                    early_stopping=True)

        #input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

        #generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)


        #preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        summary = [tokenizer.decode(output[0], skip_special_tokens=True)]
        #summary = tokenizer.decode(output[1])
        
        return jsonify ({"T5 pretrained summary": summary})
    

    @app.route('/Textsummary/extractive', methods=['GET'])
    def bert():
        CONTRACTION_MAP = {
                            "ain't": "is not",
                            "aren't": "are not",
                            "can't": "cannot",
                            "can't've": "cannot have",
                            "'cause": "because",
                            "could've": "could have",
                            "couldn't": "could not",
                            "couldn't've": "could not have",
                            "didn't": "did not",
                            "doesn't": "does not",
                            "don't": "do not",
                            "hadn't": "had not",
                            "hadn't've": "had not have",
                            "hasn't": "has not",
                            "haven't": "have not",
                            "he'd": "he would",
                            "he'd've": "he would have",
                            "he'll": "he will",
                            "he'll've": "he he will have",
                            "he's": "he is",
                            "how'd": "how did",
                            "how'd'y": "how do you",
                            "how'll": "how will",
                            "how's": "how is",
                            "I'd": "I would",
                            "I'd've": "I would have",
                            "I'll": "I will",
                            "I'll've": "I will have",
                            "I'm": "I am",
                            "I've": "I have",
                            "i'd": "i would",
                            "i'd've": "i would have",
                            "i'll": "i will",
                            "i'll've": "i will have",
                            "i'm": "i am",
                            "i've": "i have",
                            "isn't": "is not",
                            "it'd": "it would",
                            "it'd've": "it would have",
                            "it'll": "it will",
                            "it'll've": "it will have",
                            "it's": "it is",
                            "let's": "let us",
                            "ma'am": "madam",
                            "mayn't": "may not",
                            "might've": "might have",
                            "mightn't": "might not",
                            "mightn't've": "might not have",
                            "must've": "must have",
                            "mustn't": "must not",
                            "mustn't've": "must not have",
                            "needn't": "need not",
                            "needn't've": "need not have",
                            "o'clock": "of the clock",
                            "oughtn't": "ought not",
                            "oughtn't've": "ought not have",
                            "shan't": "shall not",
                            "sha'n't": "shall not",
                            "shan't've": "shall not have",
                            "she'd": "she would",
                            "she'd've": "she would have",
                            "she'll": "she will",
                            "she'll've": "she will have",
                            "she's": "she is",
                            "should've": "should have",
                            "shouldn't": "should not",
                            "shouldn't've": "should not have",
                            "so've": "so have",
                            "so's": "so as",
                            "that'd": "that would",
                            "that'd've": "that would have",
                            "that's": "that is",
                            "there'd": "there would",
                            "there'd've": "there would have",
                            "there's": "there is",
                            "they'd": "they would",
                            "they'd've": "they would have",
                            "they'll": "they will",
                            "they'll've": "they will have",
                            "they're": "they are",
                            "they've": "they have",
                            "to've": "to have",
                            "wasn't": "was not",
                            "we'd": "we would",
                            "we'd've": "we would have",
                            "we'll": "we will",
                            "we'll've": "we will have",
                            "we're": "we are",
                            "we've": "we have",
                            "weren't": "were not",
                            "what'll": "what will",
                            "what'll've": "what will have",
                            "what're": "what are",
                            "what's": "what is",
                            "what've": "what have",
                            "when's": "when is",
                            "when've": "when have",
                            "where'd": "where did",
                            "where's": "where is",
                            "where've": "where have",
                            "who'll": "who will",
                            "who'll've": "who will have",
                            "who's": "who is",
                            "who've": "who have",
                            "why's": "why is",
                            "why've": "why have",
                            "will've": "will have",
                            "won't": "will not",
                            "won't've": "will not have",
                            "would've": "would have",
                            "wouldn't": "would not",
                            "wouldn't've": "would not have",
                            "y'all": "you all",
                            "y'all'd": "you all would",
                            "y'all'd've": "you all would have",
                            "y'all're": "you all are",
                            "y'all've": "you all have",
                            "you'd": "you would",
                            "you'd've": "you would have",
                            "you'll": "you will",
                            "you'll've": "you will have",
                            "you're": "you are",
                            "you've": "you have"
                            }
        def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
            contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                                  flags=re.IGNORECASE|re.DOTALL)
            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expanded_contraction = contraction_mapping.get(match)\
                                         if contraction_mapping.get(match)\
                                         else contraction_mapping.get(match.lower())                       
                expanded_contraction = first_char+expanded_contraction[1:]
                return expanded_contraction
                    
            expanded_text = contractions_pattern.sub(expand_match, text)
            expanded_text = re.sub("'", "", expanded_text)
            return expanded_text

        def remove_special_characters(text, remove_digits=False):
            pattern = r'[^.a-zA-z0-9\s]' if remove_digits else r'[^a-zA-z\s]'
            text = re.sub(pattern,'', text)
            return text



        def normalize_corpus(doc, contraction_expansion = True, text_lower_case = True, special_char_removal = True,remove_digits=True):

            if text_lower_case:
                doc = doc.lower()

            if special_char_removal:
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = remove_special_characters(doc, remove_digits = remove_digits)
                doc = re.sub(' +',' ', doc)
            if contraction_expansion:
                doc = expand_contractions(doc)

            return doc


        data= request.form['text']
        normalize_text =normalize_corpus(data)
        sm = Summarizer()

        result = sm(body=normalize_text, ratio=0.15)
        #result = sm(body=normalize_text, num_sentences=5)
        result = '\n'.join(nltk.sent_tokenize(result))
        return jsonify ({"BERT summary": result})

    @app.route('/Textsummary/textrank', methods=['GET'])
    def textrank():
        CONTRACTION_MAP = {
                            "ain't": "is not",
                            "aren't": "are not",
                            "can't": "cannot",
                            "can't've": "cannot have",
                            "'cause": "because",
                            "could've": "could have",
                            "couldn't": "could not",
                            "couldn't've": "could not have",
                            "didn't": "did not",
                            "doesn't": "does not",
                            "don't": "do not",
                            "hadn't": "had not",
                            "hadn't've": "had not have",
                            "hasn't": "has not",
                            "haven't": "have not",
                            "he'd": "he would",
                            "he'd've": "he would have",
                            "he'll": "he will",
                            "he'll've": "he he will have",
                            "he's": "he is",
                            "how'd": "how did",
                            "how'd'y": "how do you",
                            "how'll": "how will",
                            "how's": "how is",
                            "I'd": "I would",
                            "I'd've": "I would have",
                            "I'll": "I will",
                            "I'll've": "I will have",
                            "I'm": "I am",
                            "I've": "I have",
                            "i'd": "i would",
                            "i'd've": "i would have",
                            "i'll": "i will",
                            "i'll've": "i will have",
                            "i'm": "i am",
                            "i've": "i have",
                            "isn't": "is not",
                            "it'd": "it would",
                            "it'd've": "it would have",
                            "it'll": "it will",
                            "it'll've": "it will have",
                            "it's": "it is",
                            "let's": "let us",
                            "ma'am": "madam",
                            "mayn't": "may not",
                            "might've": "might have",
                            "mightn't": "might not",
                            "mightn't've": "might not have",
                            "must've": "must have",
                            "mustn't": "must not",
                            "mustn't've": "must not have",
                            "needn't": "need not",
                            "needn't've": "need not have",
                            "o'clock": "of the clock",
                            "oughtn't": "ought not",
                            "oughtn't've": "ought not have",
                            "shan't": "shall not",
                            "sha'n't": "shall not",
                            "shan't've": "shall not have",
                            "she'd": "she would",
                            "she'd've": "she would have",
                            "she'll": "she will",
                            "she'll've": "she will have",
                            "she's": "she is",
                            "should've": "should have",
                            "shouldn't": "should not",
                            "shouldn't've": "should not have",
                            "so've": "so have",
                            "so's": "so as",
                            "that'd": "that would",
                            "that'd've": "that would have",
                            "that's": "that is",
                            "there'd": "there would",
                            "there'd've": "there would have",
                            "there's": "there is",
                            "they'd": "they would",
                            "they'd've": "they would have",
                            "they'll": "they will",
                            "they'll've": "they will have",
                            "they're": "they are",
                            "they've": "they have",
                            "to've": "to have",
                            "wasn't": "was not",
                            "we'd": "we would",
                            "we'd've": "we would have",
                            "we'll": "we will",
                            "we'll've": "we will have",
                            "we're": "we are",
                            "we've": "we have",
                            "weren't": "were not",
                            "what'll": "what will",
                            "what'll've": "what will have",
                            "what're": "what are",
                            "what's": "what is",
                            "what've": "what have",
                            "when's": "when is",
                            "when've": "when have",
                            "where'd": "where did",
                            "where's": "where is",
                            "where've": "where have",
                            "who'll": "who will",
                            "who'll've": "who will have",
                            "who's": "who is",
                            "who've": "who have",
                            "why's": "why is",
                            "why've": "why have",
                            "will've": "will have",
                            "won't": "will not",
                            "won't've": "will not have",
                            "would've": "would have",
                            "wouldn't": "would not",
                            "wouldn't've": "would not have",
                            "y'all": "you all",
                            "y'all'd": "you all would",
                            "y'all'd've": "you all would have",
                            "y'all're": "you all are",
                            "y'all've": "you all have",
                            "you'd": "you would",
                            "you'd've": "you would have",
                            "you'll": "you will",
                            "you'll've": "you will have",
                            "you're": "you are",
                            "you've": "you have"
                            }
        def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
            contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
            def expand_match(contraction):
                 match = contraction.group(0)
                 first_char = match[0]
                 expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
                 expanded_contraction = first_char+expanded_contraction[1:]
                 return expanded_contraction
        
            expanded_text = contractions_pattern.sub(expand_match, text)
            expanded_text = re.sub("'", "", expanded_text)
            return expanded_text

        def remove_special_characters(text, remove_digits=False):
            pattern = r'[^.a-zA-z0-9\s]' if remove_digits else r'[^a-zA-z\s]'
            text = re.sub(pattern,'', text)
            return text



        def normalize_corpus(doc, contraction_expansion = True, text_lower_case = True, special_char_removal = True,remove_digits=True):

            if text_lower_case:
                doc = doc.lower()

            if special_char_removal:
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = remove_special_characters(doc, remove_digits = remove_digits)
                doc = re.sub(' +',' ', doc)
            if contraction_expansion:
                doc = expand_contractions(doc)

            return doc

        t_lst = []
        s_lst = []
        data= request.form['text']
        normalize_text =normalize_corpus(data)

        List=[]
        finalsum=[]
        tsentences=sent_tokenize(normalize_text)
        List.append(tsentences)

        str =''
        for items in List:
            tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
            dt_matrix = tv.fit_transform(items)
            dt_matrix = dt_matrix.toarray()

            vocab = tv.get_feature_names()
            td_matrix = dt_matrix.T
            pd.DataFrame(np.round(td_matrix, 2), index=vocab).head(10)

            similarity_matrix=np.matmul(np.array(dt_matrix),np.array(dt_matrix.T))
            similarity_graph=nx.from_numpy_array(similarity_matrix)
            scores=nx.pagerank(similarity_graph)
            ranked_sentences=sorted(((score,index) for index,score in scores.items()),reverse=True)
            top_sentence_indices=[ranked_sentences[index][1] for index in range(3)]
            top_sentence_indices.sort()
            str=(''.join(np.array(items)[top_sentence_indices]))
            print(str)
        
        return jsonify ({"Textrank summary": str})
        
        
        
#api.add_resource(Textsummarization, '/Textsummary/v2') 
                 
if __name__== "__main__":
   app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4444)),debug='True')