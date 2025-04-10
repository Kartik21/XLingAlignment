import os
import time
import torch
from dataset_utils.FLORES200 import FLORES200
from dataset_utils.xCOPA import XCOPA
from dataset_utils.xStoryCloze import XSTORYCLOZE
from dataset_utils.Belebele import BELEBELE
import json
from datasets import load_dataset #load_dataset from Huggingface

class DataSetLoader:

    '''
    Load the dataset based on the dataset name and the list of languages. 
    Currently set up for Belebele, Flores, XCOPA, and XNLI but can be expanded.
    Used by LLM_generation to load the datasets.
    '''

    def __init__(self, args):
        dataset_dict = {'flores': FLORES200,
                        'xcopa': XCOPA,
                        'xstorycloze': XSTORYCLOZE,
                        'belebele': BELEBELE} 
        self.dataset = dataset_dict[args.dataset]()

        '''
        For every other dataset with the exception of xcopa, the source lang or the target lang is just english depending on the translation direction.
        The below if condition assigns lang list based on the translation direction.
        The only exception (XCOPA) does not have a single langlist, so it is assigned in the get_xcopa function.
        '''

        if args.dataset != 'xcopa':
            if args.translation_task == 'entoxx':
                self.langlist = args.targetlang
            else:
                self.langlist = args.sourcelang

    def get_xcopa(self, args):
        data = {}
        question = {}
        label = {}
        idx = {}
        if args.translation_task == 'entoxx':
            for lang in args.targetlang:
                sourcelang = 'english_'+lang
                data[sourcelang], question[sourcelang], label[sourcelang], idx[sourcelang] = self.dataset.get_dataset(args,self.dataset.langkeys[sourcelang])
                data[lang], question[lang], label[lang], idx[lang] = self.dataset.get_dataset(args,self.dataset.langkeys[lang])
            

        elif args.translation_task == 'xxtoen':
            for lang in args.sourcelang:      
                data[lang], question[lang], label[lang], idx[lang] = self.dataset.get_dataset(args,self.dataset.langkeys[lang])
                targetlang = 'english_'+lang
                data[targetlang], question[targetlang], label[targetlang], idx[targetlang] = self.dataset.get_dataset(args,self.dataset.langkeys[targetlang])
        return data, question, label, idx
    

    def get_xstorycloze(self,args):
        data = {}
        label = {}
        idx = {}
        
        for lang in self.langlist:
            data[lang], label[lang], idx[lang] = self.dataset.get_dataset(args,self.dataset.langkeys[lang])
    
        data['english'], label['english'], idx['english'] = self.dataset.get_dataset(args,self.dataset.langkeys['english'])
        return data, label, idx
    
    def get_flores(self, args):
        data = {}
        
        for lang in self.langlist:
            data[lang] = self.dataset.get_dataset(args, self.dataset.langkeys[lang])    
        data['english'] = self.dataset.get_dataset(args,self.dataset.langkeys['english'])
        return data
    
    def get_belebele(self, args):
        data = {}
        key = {}
        for lang in self.langlist:
            data[lang], key[lang] = self.dataset.get_dataset(args,self.dataset.langkeys[lang])
        data['english'], key['english'] = self.dataset.get_dataset(args,self.dataset.langkeys['english'])
        return data, key
    
    def get_xnli(self, args):
        data = {}
        label = {}
        idx = {}
        for lang in self.langlist:
            data[lang], label[lang], idx[lang] = self.dataset.get_dataset(args,self.dataset.langkeys[lang])
        data['english'], label['english'], idx['english'] = self.dataset.get_dataset(args,self.dataset.langkeys['english'])
        return data, label, idx
    
class JsonHandler:
    '''
    This class takes care of dumping the results of the LLM generation to a json file so that it enables future analysis.
    '''

    def __init__(self, save_dir, data):
        self.save_dir = save_dir
        self.input_data = data

    def write_json(self, args, lang, responses, primary_key):
        
        #print(responses)
        output_list = []
        if args.translation_task == 'entoxx':
            targetlang = lang
            sourcelang = 'english' if args.dataset!= 'xcopa' else 'english_'+lang

        else:
            sourcelang = lang
            targetlang = 'english' if args.dataset!= 'xcopa' else 'english_'+lang

        if args.dataset != 'belebele':
            for i in range(len(self.input_data[sourcelang]) if args.limit is None else args.limit):
                output_list.append({'src': self.input_data[sourcelang][i], 'mt': responses[i].strip(), 'ref': self.input_data[targetlang][i]})
        else:
            if args.translation_task == 'entoxx':
                for i in range(len(self.input_data['english']) if args.limit is None else args.limit):
                    for j in range(len(primary_key[lang])):
                        if primary_key[lang][j] == primary_key['english'][i]:
                            output_list.append({'src': self.input_data['english'][i], 'mt': responses[i].strip(), 'ref': self.input_data[lang][j]})
            else:
                for i in range(len(self.input_data[lang]) if args.limit is None else args.limit):
                    for j in range(len(primary_key['english'])):
                        if primary_key['english'][j] == primary_key[lang][i]:
                            output_list.append({'src': self.input_data[lang][i], 'mt': responses[i].strip(), 'ref': self.input_data['english'][j]})


        if args.translation_task == 'entoxx':
            file_name = f"english_{args.llm_name}--{lang}.json"
        else:
            file_name = f"{lang}--{args.llm_name}--english.json"
            
        with open(self.save_dir + file_name, 'w', encoding='utf8') as output_file:
            json.dump(output_list, output_file, ensure_ascii=False)

class ContextBuilder:

    
    def __init__(self, translation_task, targetlang):
        self.translation_task = translation_task
        self.targetlang = targetlang

    def build_context(self, num_fewshot, key, context_dataset='flores'):
        '''
        This function builds the context for the few-shot learning. It loads the dataset and returns the context.
        '''
        
        #This is for xstorycloze and xcopa datasets which have arabic and chinese splits. 
        #Since we use FLORES-200 to create the in-context examples, we use the simplified chinese (zho_Hans) and modern standard arabc (arb_Arab) to create ICL examples.

        if self.targetlang == 'arabic':
            targetlang_formatted = 'arb_Arab'

        if self.targetlang == 'chinese':
            targetlang_formatted = 'zho_Hans'
        
        else:

            #Otherwise, targetlang is directly derived from FLORES
            targetlang_formatted = FLORES200().langkeys[self.targetlang]

        dataset = load_dataset("facebook/flores", 'all', trust_remote_code=True)
        #Use dev dataset to create ICL examples
        dev_dataset = dataset['dev']
        
        context = ""

        if self.translation_task == 'entoxx':
            count = 0
            i = 0
            while count < num_fewshot:
                url = dev_dataset[i]['URL']
                if url != key:
                    context += "English: " + dev_dataset[i]['sentence_eng_Latn'] + "\n" + self.targetlang.capitalize() + ":" + dev_dataset[i]['sentence_'+targetlang_formatted] + "\n\n"
                    count = count + 1
                i = i + 1
        
        else:
            count = 0
            i = 0
            while count < num_fewshot:
                url = dev_dataset[i]['URL']
                if url != key:
                    context += self.targetlang.capitalize() + ": " + dev_dataset[i]['sentence_'+targetlang_formatted] + "\n" + "English:" + dev_dataset[i]['sentence_eng_Latn'] + "\n\n"
                    count = count + 1
                i = i + 1
        
        
        return context











        


                    
                        




                    








    

    


             
        








    







    



        



        






        

    






    

        
