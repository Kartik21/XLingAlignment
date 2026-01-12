import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import argparse
from collections import defaultdict
from datasets import load_dataset #load_dataset from Huggingface

def load_correct_answer(args):
    lang_dict = {'arabic': 'ar', 
                 'english': 'en', 
                 'spanish': 'es',
                 'basque': 'eu',
                 'indonesian': 'id',
                 'burmese': 'my',
                 'russian': 'ru',
                 'telugu': 'te',
                 'chinese': 'zh',
                 'swahili': 'sw',
                 'hindi': 'hi'}
    lang_code = lang_dict[args.lang]
    hf = load_dataset("juletxara/xstory_cloze", lang_code)
    answer_right_ending = hf['eval']['answer_right_ending']
    answer = []
    for i in range(len(answer_right_ending)):
        answer.append(answer_right_ending[i])
    return answer

def compute_alignment(args):

    ''' 
    Load embeddings
    
    '''
    layer_num = 32 
    with open(os.path.join(args.embedding_path, "english_choice1.pkl"), "rb") as pickle_file:
        english_choice1 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, "english_choice2.pkl"), "rb") as pickle_file:
        english_choice2 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, f"{args.lang}_choice1.pkl"), "rb") as pickle_file:
        lang_choice1 = pickle.load(pickle_file)
    
    with open(os.path.join(args.embedding_path, f"{args.lang}_choice2.pkl"), "rb") as pickle_file:
        lang_choice2 = pickle.load(pickle_file)


    def cosine_similarity(array1, array2):
        cosine_dist = cosine(array1, array2)
        cosine_similarity = 1 - cosine_dist
        return cosine_similarity
    

    eng_choice1_formatted_lasttoken = defaultdict(dict)
    eng_choice2_formatted_lasttoken = defaultdict(dict)
    lang_choice1_formatted_lasttoken = defaultdict(dict)
    lang_choice2_formatted_lasttoken = defaultdict(dict)
    


    binary_alignment_matrix_lasttoken = defaultdict(dict)



    
    # Compute alignment per layer for each sentence
    for layer in range(layer_num):  # Iterate over layers
        for item in english_choice1[layer]:    
            eng_choice1_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            

        for item in english_choice2[layer]:
            eng_choice2_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            

        for item in lang_choice1[layer]:
            lang_choice1_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            
    
        for item in lang_choice2[layer]:
            lang_choice2_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            


    for layer in range(layer_num):
        for idx in range(1511):
            cs_11_lasttoken= cosine_similarity(eng_choice1_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_22_lasttoken= cosine_similarity(eng_choice2_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            
            cs_12_lasttoken = cosine_similarity(lang_choice1_formatted_lasttoken[layer][idx+1], eng_choice2_formatted_lasttoken[layer][idx+1])
            cs_21_lasttoken = cosine_similarity(lang_choice2_formatted_lasttoken[layer][idx+1], eng_choice1_formatted_lasttoken[layer][idx+1])

            cs_12_lasttoken_within_lang = cosine_similarity(lang_choice1_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_12_lasttoken_within_eng = cosine_similarity(eng_choice1_formatted_lasttoken[layer][idx+1], eng_choice2_formatted_lasttoken[layer][idx+1])
           

            

            if (cs_11_lasttoken> cs_12_lasttoken_within_eng) & (cs_22_lasttoken> cs_12_lasttoken_within_eng) & (cs_11_lasttoken > cs_12_lasttoken_within_lang) & (cs_22_lasttoken> cs_12_lasttoken_within_lang) & (cs_11_lasttoken > cs_12_lasttoken) & (cs_11_lasttoken > cs_21_lasttoken) & (cs_22_lasttoken > cs_12_lasttoken) & (cs_22_lasttoken > cs_21_lasttoken):
                binary_alignment_matrix_lasttoken[idx][layer] = 1
            else:
                binary_alignment_matrix_lasttoken[idx][layer] = 0
                

          
              
    binary_dict_data_lasttoken = {k: dict(v) for k, v in binary_alignment_matrix_lasttoken.items()}

    os.makedirs(args.save_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Write to JSON file
    
    with open(args.save_path+f'{args.lang}_lasttoken.json', 'w') as f:
        json.dump(binary_dict_data_lasttoken, f, indent=4)


    
    
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Process Arguments for experiments with the selected LLM on various datasets')
     parser.add_argument('--llm_name', type=str, help='LLM name')
     parser.add_argument('--lang', type=str, help = 'language')
     parser.add_argument('--save_path', type=str)
     parser.add_argument('--embedding_path', type=str)
     args = parser.parse_args()

     compute_alignment(args)


