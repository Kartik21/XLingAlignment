import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import argparse
from collections import defaultdict

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import argparse
from collections import defaultdict


def compute_alignment(args):

    with open(os.path.join(args.embedding_path, f"english{args.lang}_choice1.pkl"), "rb") as pickle_file:
        english_choice1 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, f"english{args.lang}_choice2.pkl"), "rb") as pickle_file:
        english_choice2 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, f"{args.lang}_choice1.pkl"), "rb") as pickle_file:
        lang_choice1 = pickle.load(pickle_file)
    
    with open(os.path.join(args.embedding_path, f"{args.lang}_choice2.pkl"), "rb") as pickle_file:
        lang_choice2 = pickle.load(pickle_file)


    def cosine_similarity(array1, array2):

        cosine_dist = cosine(array1, array2)
        cosine_similarity = 1 - cosine_dist
        return cosine_similarity
    

    english_choice1_formatted_lasttoken = defaultdict(dict)
    english_choice2_formatted_lasttoken = defaultdict(dict)
    lang_choice1_formatted_lasttoken = defaultdict(dict)
    lang_choice2_formatted_lasttoken = defaultdict(dict)

    binary_alignment_matrix_lasttoken = defaultdict(dict)



    
    # Compute alignment per layer for each sentence
    for layer in range(32):  # Iterate over layers
        for item in english_choice1[layer]:    
            english_choice1_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
           
        for item in english_choice2[layer]:
            english_choice2_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            
        for item in lang_choice1[layer]:
            lang_choice1_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
  
        for item in lang_choice2[layer]:
            lang_choice2_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
  


    for layer in range(32):
        for idx in range(500):
            cs_11_lasttoken = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_22_lasttoken = cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
        
            cs_21_lasttoken = cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_12_lasttoken = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])


            cs_12_lasttoken_within_lang = cosine_similarity(lang_choice1_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_12_lasttoken_within_eng = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], english_choice2_formatted_lasttoken[layer][idx+1])
            
            mismatched_max_lasttoken = max(cs_21_lasttoken, cs_12_lasttoken,cs_12_lasttoken_within_lang,cs_12_lasttoken_within_eng)

            
            print(f'lang: {args.lang}, idx: {idx}, 11: {cs_11_lasttoken}, 22: {cs_22_lasttoken}, max: {mismatched_max_lasttoken}')
            if (cs_11_lasttoken > mismatched_max_lasttoken) & (cs_22_lasttoken > mismatched_max_lasttoken):
                binary_alignment_matrix_lasttoken[idx][layer] = 1
            else:
                binary_alignment_matrix_lasttoken[idx][layer] = 0
   
    binary_dict_data_lasttoken = {k: dict(v) for k, v in binary_alignment_matrix_lasttoken.items()}
    os.makedirs(args.save_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Write to JSON file    
    with open(args.save_path+f'DALI_{args.lang}_lasttoken.json', 'w') as f:
        json.dump(binary_dict_data_lasttoken, f, indent=4)

    
    
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Process Arguments for experiments with the selected LLM on various datasets')
     parser.add_argument('--llm_name', type=str, default="Llama3.1", help='LLM name')
     parser.add_argument('--lang', type=str, default = 'estonian', help = 'language')
     parser.add_argument('--save_path', type=str, default='../alignment_outputs/Llama3.1/xcopa_dali_strong_new/')
     parser.add_argument('--embedding_path', type=str,default='/fs/nexus-scratch/kravisan/embeddings/Llama3.1/xcopa_dali/')
     args = parser.parse_args()

     compute_alignment(args)


