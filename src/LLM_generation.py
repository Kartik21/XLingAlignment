import os
import time
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, logging
from model_utils.CausalLM import CAUSAL_LM
from study_utils.log_utils import Logger
from study_utils.time_utils import elapsed_from_str, Progress
import json
import argparse
from metric_utils.COMET import COMET
from dataset_utils.DataLoader import DataSetLoader
from dataset_utils.DataLoader import JsonHandler
import numpy as np



class LLMExperiment:
    def __init__(self, logger, save_dir):
        self.logger = logger
        self.save_dir = save_dir
        self.progress = Progress(logger=logger)
        
    def load_dataset(self, args):
        self.data_loader = DataSetLoader(args)
        self.data = {}
        self.primarykey = {}
        if args.dataset == 'xcopa':
            
            self.question = {}
            self.label = {}
            self.idx = {}
            time_load_start = time.time()            
            self.data, self.question, self.label, self.idx = self.data_loader.get_xcopa(args)
            self.logger.log(f"Data loaded in time {elapsed_from_str(time_load_start)}")

        if args.dataset == 'xstorycloze':
            self.data = {}
            self.label = {}
            self.idx = {}
            time_load_start = time.time()
            self.data, self.label, self.idx = self.data_loader.get_xstorycloze(args)
            self.logger.log(f"Data loaded in time {elapsed_from_str(time_load_start)}")
                            
        if args.dataset == 'flores':
            
            time_load_start = time.time()
            self.data = self.data_loader.get_flores(args)
            self.logger.log(f"Data loaded in time {elapsed_from_str(time_load_start)}")

        if args.dataset == 'belebele':
            
            time_load_start = time.time()
            self.data, self.primarykey = self.data_loader.get_belebele(args)
            self.logger.log(f"Data loaded in time {elapsed_from_str(time_load_start)}")

       

    def run_experiment(self, model, args, save_dir, checkpoint):
        jsonhandler = JsonHandler(save_dir, self.data)
        langlist = args.targetlang if args.translation_task == 'entoxx' else args.sourcelang
        self.consolidated_dict = {}
        responses = {}
        
    
        for lang in langlist:
            #Set source and targetlang accordingly depending on translation direction
            if args.translation_task == 'entoxx':        
                targetlang = lang
                sourcelang = 'english' if args.dataset!= 'xcopa' else 'english_'+lang
            else:
                sourcelang = lang
                targetlang = 'english' if args.dataset!= 'xcopa' else 'english_'+lang
            
            self.progress.start()
            input_strings = []
            time_generation_start = time.time()
            for i in tqdm(range(0,len(self.data[sourcelang]))):
                input_strings.append(self.data[sourcelang][i])
            
            #Generate translations, primarykey is only used for Belebele as it is also based on the FLORES dataset. 
            responses[lang] = model.generate(args, input_strings, self.primarykey[sourcelang] if args.dataset=='belebele' else [], lang)
            self.logger.log(f"Finished translations for {lang} in {args.dataset} using {args.llm_name}")
            self.logger.log(f"Data generated in time {elapsed_from_str(time_generation_start)}")
            jsonhandler.write_json(args, lang, responses[lang], self.primarykey)


 
    def run_COMET(self, args, save_dir):
        self.cometKIWIscore = {}
        self.cometscore = {}
        self.cometscore_sample = {}
        self.cometKIWIscore_sample = {}
        comet = COMET()

        #Gets the COMET score for the generated translations
        if args.translation_task == 'entoxx':
            for lang in args.targetlang:

                file_name = file_name = save_dir+f"english_{args.llm_name}--{lang}.json" 
                with open(file_name, encoding="utf-8") as f:
                    template = json.load(f)

                template = [{'src': item['src'], 'mt': item['mt'].strip(), 'ref': item['ref']} for item in template] 
                #Generates the COMET score for each sample
                self.cometscore_sample[lang] = comet.run_COMET(template)
                self.cometKIWIscore_sample[lang] = comet.run_COMETKIWI(template)

                #Generates the overall COMET score
                self.cometscore[lang] = (self.cometscore_sample[lang]['system_score'], np.median(self.cometscore_sample[lang]['scores']))
                self.cometKIWIscore[lang] = (self.cometKIWIscore_sample[lang]['system_score'], np.median(self.cometKIWIscore_sample[lang]['scores']))
                
                
                
            file_name = f"entoxx_{args.llm_name}_{args.dataset}"    
            with open(save_dir + file_name+'_COMETKIWI.json', 'w', encoding='utf8') as output_file:
                json.dump(self.cometKIWIscore, output_file, ensure_ascii=False)
            
            with open(save_dir + file_name+'_COMETKIWI_sample.json', 'w', encoding='utf8') as output_file:
                json.dump(self.cometKIWIscore_sample, output_file, ensure_ascii=False)
            
            with open(save_dir + file_name+'_COMET_sample.json', 'w', encoding='utf8') as output_file:
                json.dump(self.cometscore_sample, output_file, ensure_ascii=False)
            
            with open(save_dir + file_name+'_COMET.json', 'w', encoding='utf8') as output_file:
                json.dump(self.cometscore, output_file, ensure_ascii=False)


        if args.translation_task == 'xxtoen':
            for lang in args.sourcelang:

                file_name = save_dir+f"{lang}--{args.llm_name}--english.json"
                with open(file_name, encoding="utf-8") as f:
                    template = json.load(f)
                
                template = [{'src': item['src'], 'mt': item['mt'].strip(), 'ref': item['ref']} for item in template]
                self.cometscore_sample[lang] = comet.run_COMET(template)
                self.cometKIWIscore_sample[lang] = comet.run_COMETKIWI(template)
                
                self.cometscore[lang] = (self.cometscore_sample[lang]['system_score'], np.median(self.cometscore_sample[lang]['scores']))
                self.cometKIWIscore[lang] = self.cometKIWIscore_sample[lang]['system_score'], np.median(self.cometKIWIscore_sample[lang]['scores'])
                self.logger.log(f"COMET score for {lang} is equal to: {self.cometscore[lang]*100}")
                self.logger.log(f"COMETKIWI score for {lang} is equal to: {self.cometKIWIscore[lang]*100}")
            
            file_name = f"xxtoen_{args.llm_name}_{args.dataset}"
            with open(save_dir + file_name+'_COMETKIWI.json', 'w', encoding='utf8') as output_file:
                json.dump(self.cometKIWIscore, output_file, ensure_ascii=False)


            with open(save_dir + file_name+'_COMETKIWI_sample.json', 'w', encoding='utf8') as output_file:
                json.dump(self.cometKIWIscore_sample, output_file, ensure_ascii=False)


            with open(save_dir + file_name+'_COMET.json', 'w', encoding='utf8') as output_file:
                json.dump(self.cometscore, output_file, ensure_ascii=False)

            with open(save_dir + file_name+'_COMET_sample.json', 'w', encoding='utf8') as output_file:
                json.dump(self.cometscore_sample, output_file, ensure_ascii=False)

      
        
    def terminate_and_save(self):
        self.logger.log("Terminating the experiment")
        self.logger.log("Experiment saved successfully")

    

if __name__ == '__main__':

    def list_of_strings(args):
        return args.split(',')
    # Step 2: Load model and tokenizer
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with the selected LLM on various datasets')
    parser.add_argument('--llm_name', type=str, default="Aya23", help='LLM name')
    parser.add_argument('--sourcelang', type=list_of_strings, help = 'Source language')
    parser.add_argument('--targetlang', type=list_of_strings, help = 'List of target languages')
    parser.add_argument('--translation_task', type=str, default = 'entoxx', help = 'Translation task, entoXX or XXtoen')
    parser.add_argument('--save_dir', type = str, default = "../translation_outputs")
    parser.add_argument('--strata', type = str, default = "dev", help="Choose between dev or test")
    parser.add_argument('--num_fewshot', type=int, default = 5, help="number of in-context examples for non-instruction tuned models")
    parser.add_argument('--dataset', type = str, default = "flores", help="Choose between flores or xcopa")
    parser.add_argument('--combination_approach', type = str, default = "concatenation", help="Choose between concatenation or separate_inputs; Not used for FLORES dataset")    
    parser.add_argument('--max_tokens', type = int, default = 512, help="Choose a max_new_token for generation")
    parser.add_argument('--do_sample', type = bool, default = False, help="Choose whether to sample or not during generation")
    parser.add_argument('--quantize_4bit', type = bool, default = True, help="Choose if you want 4 bit quantization")
    parser.add_argument('--limit', type = int, default = None, help="Choose the number of examples to generate")
    parser.add_argument('--input_field', type = str, default = None, help = "Choose the input field of the dataset")
    parser.add_argument('--translation_mode', type= str, default='translation', help="Choose between translation and evaluation")
    args = parser.parse_args()
    
    save_dir = args.save_dir
    
    modified_save_dir = f"{save_dir}/{args.llm_name}/{args.dataset}_100/{args.input_field}/"
    if not os.path.exists(modified_save_dir):
        os.makedirs(modified_save_dir, exist_ok=True)
    logger = Logger(save_dir=modified_save_dir, fname=f"{args.llm_name}-log-{args.sourcelang}-{args.targetlang}.txt")

    # Step 4: Create an experiment

    checkpoint_dict = {"Aya23": "CohereForAI/aya-23-8B", "Llama3.1": "meta-llama/Llama-3.1-8B", "Gemma": "google/gemma-1.1-7b-it","Mistral": "mistralai/Mistral-7B-Instruct-v0.2", "Bloom": "bigscience/bloomz-7b1"}

    

    checkpoint = checkpoint_dict[args.llm_name]

    if args.translation_mode == 'translation':
        model = CAUSAL_LM(args, logger, checkpoint)
        experiment = LLMExperiment(logger=logger, save_dir=save_dir)
        experiment.load_dataset(args)
        experiment.run_experiment(model, args,modified_save_dir, checkpoint)
    
    if args.translation_mode == 'evaluation':
        experiment = LLMExperiment(logger=logger, save_dir=save_dir)
        experiment.run_COMET(args, modified_save_dir)
        
    experiment.terminate_and_save()
