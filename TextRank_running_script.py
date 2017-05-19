#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:07:36 2017

Applies the TextRank keyphrase exctraction model on a text file.
Prints all extracted keyphrases with their scores to the console.

Should be used the following way:
    python -m -fp 'file_path'(string)
    
Optional parameter ranges:
    -N, --number_of_keynodes: Any positive integer. Controls how many keywords are
                                considered for keyphrase creation.
    -M, --number_of_keyphrases: Any positive integer. Controls how many keyphrases are
                                extracted.
    -lamda, --inv_random_jump_probability: Any float between 0 and 1. Model parameter
    -nh, --no_handles: Flag. If flagged, twitter handles are ignored and not used
                        for keyword or keyphrase extraction.
    -ws, --window_size: Any positive integer. Controls the size of the co-occurrence
                        scanning window.

@author: bettmensch
"""
from TextRank import load_text
from TextRank import tokenize_tag
from TextRank import add_tokens_tags
from TextRank import rank_extract_keynodes
from TextRank import extract_keyphrases
from TextRank import rank_keyphrases
from networkx import MultiDiGraph as MDG
import argparse

def main():
    """Runs a test drive of the preceding functions."""
    
    parser = argparse.ArgumentParser(description='Extracts keyphrases from twitter corpus')
    
    parser.add_argument('-fp', '--file_path',
                       required = False,
                       help = 'Set the filepath for the file containing the jsonl.',
                       default = 'LOTR_3')    
    parser.add_argument('-N','--number_of_keynodes',
                        required=False,
                        help='Set number of top N keynodes according to PageRank.',
                        type=int,
                        default = 5000)
    parser.add_argument('-M','--number_of_keyphrases',
                        required=False,
                        help='Set the number of top M keyphrases considered.',
                        type = int,
                        default=100)
    parser.add_argument('-lamda','--inv_random_jump_prob',
                        required=False,
                        help='Set the random jump probability for the PageRank.',
                        type=float,
                        default=0.84)
    parser.add_argument('-nh','--no_handles',
                        required=False,
                        help='Flag if twitter handles should be omitted.',
                        action='store_true',
                        default=False)
    parser.add_argument('-ws','--window_size',
                        required=False,
                        help="Set the co-occurrence scanner's size.",
                        type=int,
                        default = 5)
    
    args = vars(parser.parse_args())
    
    file_path = args['file_path']
    N = args['number_of_keynodes']
    M = args['number_of_keyphrases']
    lamda = args['inv_random_jump_prob']
    window_size = args['window_size']
    no_handles = args['no_handles']
    
    print("Fetching 'Lord Of The Rings: The fellowship of the ring' data...")
    
    text = load_text(file_path)
        
    print("Fetched 'Lord Of The Rings: The fellowship of the ring'.")
    print("Size of textfile (in character): ", len(text))
    input("Press Enter to continue.")
    
    print("Tokenizing and tagging data...")
    
    tokens_tags = tokenize_tag(text,no_handles)
            
    print(len(tokens_tags))
            
    print("Tokenized and tagged data.")
    print("Size of tokens list (in tokens): ", len(tokens_tags))
    input("Press Enter to continue.")

    graph = MDG()
    
    print("Building graph...")
    add_tokens_tags(tokens_tags, graph, window_size)
    print("Graph built.")
    input("Press Enter to continue.")
    
    print("Running page rank...")
    top_N_tokens = rank_extract_keynodes(graph,
                                         max_iter = 50,
                                         tol = 1.0e-6,
                                         lamda = lamda,
                                         N = N)
    print("PageRank completed.")
    input("Press Enter to continue.")
    
    print("Extracting keyphrases...")
    keyphrases = extract_keyphrases(tokens_tags, top_N_tokens)
    print("Keyphrases extracted.")
    input("Press Enter to continue.")
    
    print("Ranking keyphrases...")
    top_M_keyphrases = rank_keyphrases(keyphrases, top_N_tokens, M)
    print("Keyphrases ranked.")
    input("Press Enter to continue.")
    
    for keyphrase, score in top_M_keyphrases.items():
        print("Keyphrase: ", " ".join(keyphrase), "|| Score: ", score)
    
if __name__ == "__main__":
    main()
