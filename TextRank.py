#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:11:08 2017

Contains all functions necessary to apply TextRank, the keyword extraction model,
to any text file.

@author: bettmensch
"""

from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from math import log
import itertools

SIMP_DICT = {'JJ':'ADJ', 'JJR':'ADJ', 'JJS':'ADJ', 'RBR':'ADJ', 'RBS':'ADJ',
             'VB':'VB', 'VBD':'VB', 'VBG':'VB', 'VBN':'VB', 'VBP':'VB', 'VBZ':'VB',
             'NN':'NN', 'NNP':'NN', 'NNPS':'NN', 'NNS':'NN'}
    
REL_TAG_COMBS = {('ADJ','NN'),
                  ('NN','ADJ'),
                  ('NN','NN'),
                  #('VB','NN'),
                  #('NN','VB'),
                  #('VB','ADJ'),
                  #('ADJ','VB')
                  }
    
with open('stopwords.txt', 'r') as stop_file_1:
    STOPWORDS = [line.strip() for line in stop_file_1]
    
def no_newlines(text):
    """Takes a string and deletes all newlines "\n" terms from it for better
    processing.
    Returns cleaned text."""
    clean_text = ""
    
    for character in text:
        if character != "\n":
            clean_text += character

    return clean_text
    
    
def load_text(filename):
    """Takes a filename of a text file. Loads and stores the file content in a string.
    Returns that string."""
    
    with open(filename, 'r') as text_file:
        text = text_file.read().strip()
        
        text = no_newlines(text)
        
    return text

def tokenize_tag(text,strip_handles = False, reduce_len = False, preserve_case = True):
    """Takes a text (type string) and tokenizes and tags the text.
    Takes a no_handles flag (type bool) which influences the tokenization.
    Returns an iterable of tuples (token_tag), where tags are already refined."""

    tokenizer = TweetTokenizer(strip_handles = strip_handles,
                               reduce_len = reduce_len,
                               preserve_case = preserve_case)
    
    tokens = tokenizer.tokenize(text)
    tokens_raw_tags = pos_tag(tokens)
    tokens_tags = refine_tags(tokens_raw_tags)
    
    return tokens_tags
    
def refine_tags(tokens_raw_tags,simp_dict = SIMP_DICT, stopwords = STOPWORDS):
    """Takes an iterable of tuples (token, raw_tag).
    Simplifies tags to base categories using a simplification dict
    Marks stopwords and non-base categories as 'IRR' using a stopwords list.
    Returns an iterable of tuples (token, refine_tag)."""
    
    tokens_tags = []
                                  
    for token, raw_tag in tokens_raw_tags:
        if raw_tag in SIMP_DICT.keys():
            tag = SIMP_DICT[raw_tag]
        else:
            tag = 'IRR'
        
        if token in stopwords:
            tag = 'IRR'
            
        tokens_tags.append((token, tag))
        
    return tokens_tags

def add_tokens_tags(tokens_tags, graph, window_size, rel_tag_combs = REL_TAG_COMBS):
    """Takes and iterable of tuples (token, tag).
    Takes a graph (type MultuDiGraph).
    Takes a window size (type int).
    Scans over iterable and adds relevant edges to the graph."""

        
    # set number of scanner windows based on document size and window_size
    windows_no = max(len(tokens_tags) - window_size + 1,1)
    # start adding edges and building adjacency record
    for window_index in range(windows_no):
        # create window of (token,tag) tuples
        window = tokens_tags[window_index:window_index + window_size]
        # create all combinations of (token,tag) tuples in 'window'
        all_window_combs = itertools.combinations(enumerate(window),2)
        # iterate through all combinations in 'window'
        for (i_1,(token_1,tag_1)),(i_2,(token_2,tag_2)) in all_window_combs:
            # select only those whose tag combination seems relevant, # i.e. adjective + noun
            if (tag_1,tag_2) in rel_tag_combs and token_1 != token_2:
                # integrate current co-oc into graph if necessary
                if graph.has_edge(token_1,token_2):
                    graph[token_1][token_2][0]['weight'] += 1
                else:
                    graph.add_edge(token_1,token_2,weight = 1)

def rank_extract_keynodes(graph, max_iter, tol, lamda, N):
    """Takes a graph (type MultiDiGraph).
    Takes a maximum number of iterations (type int).
    Takes a cauchy tolerance threshold (type float).
    Takes a negated random jump probability lamda \in (0,1) (type float).
    Takes the number N of top keynodes to be returned (type int).
    Runs the pagerank algorithm of the word graph
    Returns the top N nodes with their scores {node: score for node in top_N_nodes}
    (type dict)."""
    
    n = len(graph.nodes())
    # initialize outgoing weights storage
    O = {}
    
    for w_j in graph.nodes():
        score = 0
        for w_k in graph.successors(w_j):
            score += graph[w_j][w_k][0]['weight']
            
        O[w_j] = score
         
    #O = {w_j: sum([graph[w_j][w_k][0]['weight'] 
    #                for w_k in graph.successors(w_j)])
    #                    for w_j in graph.nodes()}
    
    # initialize initial node scores
    scores = {w_j: 1 / n for w_j in graph.nodes()}
    
    error = tol + 1
    iteration = 0
    
    while error >= tol and iteration <= max_iter + 1:
        temp = scores
        
        for w_i in graph.nodes():
            scores[w_i] = lamda * sum([graph[w_j][w_i][0]['weight'] / O[w_j] * temp[w_j]
                                        for w_j in graph.predecessors(w_i)]) + (1 - lamda)
    
        error = sum([abs(temp[w_i] - scores[w_i]) for w_i in graph.nodes()]) / n
        iteration += 1
    
    if error >= tol:
        # make this a warning
        print("The ranking algorithm failed to converge in the given maximum iterations.")
    else:
        # make this an info
        print("The ranking algorithm converged.")
        
    top_N_scores = {w_i: score_w_i for w_i, score_w_i in sorted(scores.items(),
                                                               key = lambda item: item[1],
                                                               reverse = True)[:N]}
    
    return top_N_scores

def extract_keyphrases(tokens_tags, top_N_scores, rel_tag_combs = REL_TAG_COMBS):
    """Takes an iterable of tuples (token, tag).
    Takes a dictionary of top_N_scores {token: token_score} keyed by token.
    Returns an iterable of keyphrases (type set) obtained from tokens_tags co-ocurrences."""
    
    keyphrases = []
    keyphrase = []
    last_tag = None

    for token, tag in tokens_tags:
        if token in top_N_scores.keys():
            if keyphrase:
                if (last_tag, tag) in rel_tag_combs:
                    keyphrase.append(token)
                    last_tag = tag
            else:
                keyphrase = [token,]
                last_tag = tag
        else:
            if keyphrase:
                #print("The type of keyphrases is ", type(keyphrases))
                keyphrases.append(tuple(keyphrase))
                keyphrase = []
                last_tag = None
        
    return set(keyphrases)

def rank_keyphrases(keyphrases, top_N_scores, M):
    """Takes an iterable of keyphrases (type set).
    Takes a dictionary of node scores {keyword: score}.
    Takes a maximum number M of keyphrases to be returned (type int).
    Calculates the keyphrase ranks based on the keywords ranks.
    Returns top_M keyphrases with their scores {phrase: score for phrase in top_M_phrases}
    (type dict)."""
    
    scores = {keyphrase: sum([top_N_scores[token] / len(keyphrase) for token in keyphrase])
                            for keyphrase in keyphrases}
    
    top_M_scores = {k_p: score for k_p, score in sorted(scores.items(),
                                                        key = lambda item: item[1],
                                                        reverse = True)[:M]}
    
    return top_M_scores