from datasets import load_dataset #load_dataset from Huggingface
from dataset_utils.abstractdataset import AbstractDataset 
from collections import defaultdict

'XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning, Ponti et al., 2020'
'Task category : Causal Reasoning'



class XCOPA(AbstractDataset):
    def __init__(self):
        super(XCOPA, self).__init__()
        self.langkeys = {'estonian': 'et',
                         'haitian creole': 'ht',
                         'indonesian': 'id',
                         'italian': 'it',
                         'ayacucho quechua': 'qu',
                         'swahili': 'sw',
                         'tamil': 'ta',
                         'thai': 'th',
                         'turkish': 'tr',
                         'vietnamese': 'vi',
                         'chinese': 'zh',



                         'english_estonian': 'translation-et',
                         'english_haitian creole': 'translation-ht',
                         'english_indonesian': 'translation-id',
                         'english_italian': 'translation-it',
                         'english_ayacucho quechua': 'translation-qu',
                         'english_swahili': 'translation-sw',
                         'english_tamil': 'translation-ta',
                         'english_thai': 'translation-th',
                         'english_turkish': 'translation-tr',
                         'english_vietnamese': 'translation-vi',
                         'english_chinese': 'translation-zh'
                         }



    def get_dataset(self, args, lang):

        '''
        Loads the xcopa dataset for both source and target languages

        args: arguments from the main function
        args.strata = dev or test 
        args.combination_approach = concatenation or separate_inputs
        '''        

        # Load the xcopa dataset
        dataset = load_dataset("xcopa", lang)
        dataset = dataset[args.strata]
        
        # Get the dataset for the source and target language
    

        dataset_premise = dataset['premise']
        dataset_choice1 = dataset['choice1']
        dataset_choice2 = dataset['choice2']
        dataset_question = dataset['question']
        dataset_label = dataset['label']
        dataset_idx = dataset['idx']

        assert len(dataset_premise) == len(dataset_choice1) == len(dataset_choice2)


        if args.combination_approach == "concatenation":

            '''
            if args.combination approach is concatenation, then premise, choice1 and choice2 are concatenated together for each 
            sentence in the dataset, thus keeping the number of datapoints the same. 
            '''
            data = []
           
            for i in range(len(dataset_premise)):
                data.append(dataset_premise[i] + " " + dataset_choice1[i] + " " + dataset_choice2[i])
                question.append(dataset_question[i])
                label.append(dataset_label[i])
                idx.append(dataset_idx[i])
                

        if args.combination_approach == "separate_inputs":

            '''
            if args.combination appraoch is separate_inputs, then premise, choice1 and choice2 
            '''
            data = []
            question = []
            label = []
            idx = []
            
            for i in range(len(dataset_premise)):
                if args.input_field == 'premise':
                    data.append(dataset_premise[i])
                if args.input_field == 'choice1':
                    data.append(dataset_choice1[i])
                if args.input_field == 'choice2':
                    data.append(dataset_choice2[i])
                question.append(dataset_question[i])
                label.append(dataset_label[i])
                idx.append(dataset_idx[i])


        return data, question, label, idx
    
    def get_dataset_accuracy(self, args, lang):
         # Load the xcopa dataset
        dataset = load_dataset("xcopa", lang)
        dataset = dataset[args.strata]

        # Get the dataset for the source and target language
    

        dataset_premise = dataset['premise']
        dataset_choice1 = dataset['choice1']
        dataset_choice2 = dataset['choice2']
        question = dataset['question']
        label = dataset['label']

        assert len(dataset_premise) == len(dataset_choice1) == len(dataset_choice2)
        data = []
        answer = []
        prompt_cause = "Based on the premise and two choices, select the choice that is the most likely cause of the premise."
        prompt_effect = "Based on the premise and two choices, select the choice that is the most likely effect of the premise."
        for i in range(len(dataset_premise)):
            prefix = prompt_cause if question[i] == "cause" else prompt_effect
            prompt = prefix + "\nPremise: " + dataset_premise[i] + "\n0. " + dataset_choice1[i] + "\n1. " + dataset_choice2[i] + "\nThe correct choice number is: "
            data.append(prompt)
            answer.append(label[i])
        
        return data, answer









