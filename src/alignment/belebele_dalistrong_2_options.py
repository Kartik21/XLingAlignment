import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import argparse
from collections import defaultdict

def load_worst_wrong_answers(args):
    worst_2choices = []
    accuracy_data_path = f'/nfshomes/kravisan/MultilingualLLMeval/accuracy_outputs/Llama3.1/belebele_5shot/{args.lang}/'
    jsonl_file = [f for f in os.listdir(accuracy_data_path) if f.endswith('.jsonl')][0]
    file_path = os.path.join(accuracy_data_path, jsonl_file)
    # Read the jsonl file line by line
    accuracy_results = []
    with open(file_path, 'r') as f:
        for line in f:
            accuracy_results.append(json.loads(line))
    for i in range(900):
        log_prob = [float(accuracy_results[i]['resps'][0][0][0]),float(accuracy_results[i]['resps'][1][0][0]),float(accuracy_results[i]['resps'][2][0][0]),float(accuracy_results[i]['resps'][3][0][0])]
        correct_answer_num = int(accuracy_results[i]['doc']['correct_answer_num'])
        wrong_choices = [j+1 for j in range(4) if j!=correct_answer_num-1]
        worst_2choices.append(sorted(wrong_choices, key=lambda x: log_prob[x - 1])[:2])
    return worst_2choices


def compute_alignment(args, worst_choices):

    with open(os.path.join(args.embedding_path, "eng_Latn_choice1.pkl"), "rb") as pickle_file:
        english_choice1 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, "eng_Latn_choice2.pkl"), "rb") as pickle_file:
        english_choice2 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, "eng_Latn_choice3.pkl"), "rb") as pickle_file:
        english_choice3 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, "eng_Latn_choice4.pkl"), "rb") as pickle_file:
        english_choice4 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, f"{args.lang}_choice1.pkl"), "rb") as pickle_file:
        lang_choice1 = pickle.load(pickle_file)
    
    with open(os.path.join(args.embedding_path, f"{args.lang}_choice2.pkl"), "rb") as pickle_file:
        lang_choice2 = pickle.load(pickle_file)
    
    with open(os.path.join(args.embedding_path, f"{args.lang}_choice3.pkl"), "rb") as pickle_file:
        lang_choice3 = pickle.load(pickle_file)

    with open(os.path.join(args.embedding_path, f"{args.lang}_choice4.pkl"), "rb") as pickle_file:
        lang_choice4 = pickle.load(pickle_file)


    def cosine_similarity(array1, array2):
        cosine_dist = cosine(array1, array2)
        cosine_similarity = 1 - cosine_dist
        return cosine_similarity
    

    english_choice1_formatted_lasttoken = defaultdict(dict)
    english_choice2_formatted_lasttoken = defaultdict(dict)
    english_choice3_formatted_lasttoken = defaultdict(dict)
    english_choice4_formatted_lasttoken = defaultdict(dict)


    lang_choice1_formatted_lasttoken = defaultdict(dict)
    lang_choice2_formatted_lasttoken = defaultdict(dict)
    lang_choice3_formatted_lasttoken = defaultdict(dict)
    lang_choice4_formatted_lasttoken = defaultdict(dict)

    binary_alignment_matrix_lasttoken = defaultdict(dict)

    # Compute alignment per layer for each sentence
    for layer in range(32):  # Iterate over layers
        for item in english_choice1[layer]:    
            english_choice1_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            
        for item in english_choice2[layer]:
            english_choice2_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            
        for item in english_choice3[layer]:
            english_choice3_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            
        for item in english_choice4[layer]:
            english_choice4_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            
        for item in lang_choice1[layer]:
            lang_choice1_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
               
        for item in lang_choice2[layer]:
            lang_choice2_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            
        for item in lang_choice3[layer]:
            lang_choice3_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            
        for item in lang_choice4[layer]:
            lang_choice4_formatted_lasttoken[layer][item['id']] = item['embd_lasttoken']
            

    for layer in range(32):
        for idx in range(900):

            cs_11_lasttoken = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_22_lasttoken = cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_33_lasttoken = cosine_similarity(english_choice3_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])
            cs_44_lasttoken = cosine_similarity(english_choice4_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
                        
            cs_12_lasttoken = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_21_lasttoken = cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_13_lasttoken= cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])
            cs_31_lasttoken= cosine_similarity(english_choice3_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_14_lasttoken= cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
            cs_41_lasttoken= cosine_similarity(english_choice4_formatted_lasttoken[layer][idx+1], lang_choice1_formatted_lasttoken[layer][idx+1])
            cs_23_lasttoken= cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])
            cs_32_lasttoken= cosine_similarity(english_choice3_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_24_lasttoken= cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
            cs_42_lasttoken= cosine_similarity(english_choice4_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_34_lasttoken= cosine_similarity(english_choice3_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
            cs_43_lasttoken= cosine_similarity(english_choice4_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])

            cs_12_lasttoken_within_eng = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], english_choice2_formatted_lasttoken[layer][idx+1])
            cs_13_lasttoken_within_eng = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], english_choice3_formatted_lasttoken[layer][idx+1])
            cs_14_lasttoken_within_eng = cosine_similarity(english_choice1_formatted_lasttoken[layer][idx+1], english_choice4_formatted_lasttoken[layer][idx+1])
            cs_23_lasttoken_within_eng= cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], english_choice3_formatted_lasttoken[layer][idx+1])
            cs_24_lasttoken_within_eng= cosine_similarity(english_choice2_formatted_lasttoken[layer][idx+1], english_choice4_formatted_lasttoken[layer][idx+1])
            cs_34_lasttoken_within_eng= cosine_similarity(english_choice3_formatted_lasttoken[layer][idx+1], english_choice4_formatted_lasttoken[layer][idx+1])

            cs_12_lasttoken_within_lang = cosine_similarity(lang_choice1_formatted_lasttoken[layer][idx+1], lang_choice2_formatted_lasttoken[layer][idx+1])
            cs_13_lasttoken_within_lang = cosine_similarity(lang_choice1_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])
            cs_14_lasttoken_within_lang = cosine_similarity(lang_choice1_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
            cs_23_lasttoken_within_lang= cosine_similarity(lang_choice2_formatted_lasttoken[layer][idx+1], lang_choice3_formatted_lasttoken[layer][idx+1])
            cs_24_lasttoken_within_lang= cosine_similarity(lang_choice2_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])
            cs_34_lasttoken_within_lang= cosine_similarity(lang_choice3_formatted_lasttoken[layer][idx+1], lang_choice4_formatted_lasttoken[layer][idx+1])

            if worst_choices[idx] == [1,2] or [2,1]:
                'remove all similaries associated with 1'
                max_non_aligned_lasttoken = max(cs_34_lasttoken,cs_43_lasttoken,cs_34_lasttoken_within_eng,cs_34_lasttoken_within_lang)
                if (cs_33_lasttoken > max_non_aligned_lasttoken) & (cs_44_lasttoken > max_non_aligned_lasttoken):
                    binary_alignment_matrix_lasttoken[idx][layer] = 1
                else:
                    binary_alignment_matrix_lasttoken[idx][layer] = 0

            if worst_choices[idx] == [1,3] or [3,1]:
                'remove all similaries associated with 1'
                max_non_aligned_lasttoken = max(cs_24_lasttoken,cs_42_lasttoken,cs_24_lasttoken_within_eng,cs_24_lasttoken_within_lang)
                if (cs_22_lasttoken > max_non_aligned_lasttoken) & (cs_44_lasttoken > max_non_aligned_lasttoken):
                    binary_alignment_matrix_lasttoken[idx][layer] = 1
                else:
                    binary_alignment_matrix_lasttoken[idx][layer] = 0

            if worst_choices[idx] == [1,4] or [4,1]:
                'remove all similaries associated with 1'
                max_non_aligned_lasttoken = max(cs_23_lasttoken,cs_32_lasttoken,cs_23_lasttoken_within_eng,cs_23_lasttoken_within_lang)
                if (cs_22_lasttoken > max_non_aligned_lasttoken) & (cs_33_lasttoken > max_non_aligned_lasttoken):
                    binary_alignment_matrix_lasttoken[idx][layer] = 1
                else:
                    binary_alignment_matrix_lasttoken[idx][layer] = 0

            if worst_choices[idx] == [2,3] or [3,2]:
                'remove all similaries associated with 1'
                max_non_aligned_lasttoken = max(cs_14_lasttoken,cs_41_lasttoken,cs_14_lasttoken_within_eng,cs_14_lasttoken_within_lang)
                if (cs_11_lasttoken > max_non_aligned_lasttoken) & (cs_44_lasttoken > max_non_aligned_lasttoken):
                    binary_alignment_matrix_lasttoken[idx][layer] = 1
                else:
                    binary_alignment_matrix_lasttoken[idx][layer] = 0

            if worst_choices[idx] == [2,4] or [4,2]:
                'remove all similaries associated with 1'
                max_non_aligned_lasttoken = max(cs_13_lasttoken,cs_31_lasttoken,cs_13_lasttoken_within_eng,cs_13_lasttoken_within_lang)
                if (cs_11_lasttoken > max_non_aligned_lasttoken) & (cs_33_lasttoken > max_non_aligned_lasttoken):
                    binary_alignment_matrix_lasttoken[idx][layer] = 1
                else:
                    binary_alignment_matrix_lasttoken[idx][layer] = 0

            if worst_choices[idx] == [3,4] or [4,3]:
                'remove all similaries associated with 1'
                max_non_aligned_lasttoken = max(cs_12_lasttoken,cs_21_lasttoken,cs_12_lasttoken_within_eng,cs_12_lasttoken_within_lang)
                if (cs_11_lasttoken > max_non_aligned_lasttoken) & (cs_22_lasttoken > max_non_aligned_lasttoken):
                    binary_alignment_matrix_lasttoken[idx][layer] = 1
                else:
                    binary_alignment_matrix_lasttoken[idx][layer] = 0
    
    binary_dict_data_lasttoken = {int(k)+1: dict(v) for k, v in binary_alignment_matrix_lasttoken.items()}
    os.makedirs(args.save_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Write to JSON file
    
    with open(args.save_path+f'DALI_{args.lang}_lasttoken.json', 'w') as f:
        json.dump(binary_dict_data_lasttoken, f, indent=4)
    
  
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Process Arguments for experiments with the selected LLM on various datasets')
     parser.add_argument('--llm_name', type=str, default="Llama3.1", help='LLM name')
     parser.add_argument('--lang', type=str, default = 'hin_Deva', help = 'language')
     parser.add_argument('--save_path', type=str, default='../alignment_outputs/Llama3.1/belebele_dalistrong_2option/')
     parser.add_argument('--embedding_path', type=str,default='/fs/nexus-scratch/kravisan/embeddings/Llama3.1/belebele_dali/')
     args = parser.parse_args()
     worst_choice_to_eliminate = load_worst_wrong_answers(args)
     print('Worst choice loaded')


     compute_alignment(args, worst_choice_to_eliminate)