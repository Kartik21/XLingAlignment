from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import tqdm
from dataset_utils.DataLoader import DataSetLoader
from dataset_utils.DataLoader import ContextBuilder

class CAUSAL_LM:
    def __init__(self, args, logger, checkpoint):
        if args.quantize_4bit:
            quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,)
        
        self.instruction_tuned_flag = True if checkpoint == 'CohereForAI/aya-23-8B' else False
        # Device for the experiment
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()
        logger.log("Loading Model and Tokenizer")
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="auto", )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        logger.log("Model and Tokenizer Loaded")
    
    def get_message_format(self, args, input_strings, primary_key, targetlang):
        
        '''
        Input_strings are a list of strings in a specific language. Prompts are formatted by a translation prefix.
        '''
        self.prompts = []
        #If the model is instruction tuned, add the translation prompt from the Aya model
        if self.instruction_tuned_flag:
            if args.translation_task == 'entoxx':
                
                for item in input_strings:
                    txt = "Translate from english" + " to " + targetlang+". Only provide translations and do not provide any further explanations.\nEnglish: "+ item + "\n" + targetlang.capitalize()+ ":"
                    self.prompts.append(txt)

            if args.translation_task == 'xxtoen':
                for item in input_strings:
                    txt = "Translate from " + targetlang + " to english. Only provide translations and do not provide any further explanations.\n" + targetlang.capitalize() + ": " + item + "\n" + "English: "
                    self.prompts.append(txt)
        
        
        else:
            #If the model is not instruction tuned, then we build the context based on the num_fewshot
            context_builder = ContextBuilder(args.translation_task, targetlang)
           
            if args.translation_task == 'entoxx':
                if args.dataset != 'belebele':
                    print('Building context')
                    context = context_builder.build_context(args.num_fewshot, primary_key)
                    for i in range(0,len(input_strings) if args.limit is None else args.limit):
                        txt = context + "English: " + input_strings[i] + "\n" + targetlang.capitalize() + ":"
                        self.prompts.append(txt)
                else:
                    for i in range(0,len(input_strings) if args.limit is None else args.limit):
                        key = primary_key[i][0] if primary_key!= [] else primary_key
                        context = context_builder.build_context(args.num_fewshot, key)
                        txt = context + "English: " + input_strings[i] + "\n" + targetlang.capitalize() + ":"
                        self.prompts.append(txt)

            else:
                if args.dataset != 'belebele':
                    print('Buidling context')
                    context = context_builder.build_context(args.num_fewshot, primary_key)
                    for i in range(0,len(input_strings) if args.limit is None else args.limit):
                        txt = context + targetlang.capitalize() + ": " + input_strings[i] + "\n" + "English: "
                        self.prompts.append(txt)
                else:
                    for i in range(0,len(input_strings) if args.limit is None else args.limit):
                        key = primary_key[i][0] if primary_key!= [] else primary_key
                        context = context_builder.build_context(args.num_fewshot, key)
                        txt = context + targetlang.capitalize() + ": " + input_strings[i] + "\n" + "English: "
                        self.prompts.append(txt)
            

        messages = []

        
        
        if self.instruction_tuned_flag:
            for i in range(len(self.prompts) if args.limit is None else args.limit):
                messages.append(
                    [{"role": "user", "content": self.prompts[i]}]
                )
        else:
            messages = [self.prompts[i] for i in range(len(self.prompts) if args.limit is None else args.limit)]

        return messages

    

    
    def generate(self, args, input_strings, primary_key, lang):
        '''
        This function generates a response for a given prompt. Promp
        '''
        messages = self.get_message_format(args, input_strings, primary_key, lang)
        


        responses = []
        #should loop through all len(messages)
        for i in tqdm.tqdm(range(len(messages))):
            if self.instruction_tuned_flag:
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                input_ids = self.tokenizer.apply_chat_template(messages[i],tokenize=True,add_generation_prompt=True,padding=True,return_tensors="pt",)
                input_ids = input_ids.to(self.model.device)
            
            else:
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                input_ids = self.tokenizer.encode(messages[i], padding=True, return_tensors="pt")
                input_ids = input_ids.to(self.model.device) 
       
            prompt_padded_len = len(input_ids[0])

            if args.do_sample:
                gen_tokens = self.model.generate(input_ids,temperature=args.temperature,top_p=args.top_p,top_k=args.top_k,max_new_tokens=args.max_tokens,do_sample=True,)
            else:
                gen_tokens = self.model.generate(input_ids,max_new_tokens=args.max_tokens,do_sample=False,early_stopping=True,stop_strings="\n",tokenizer=self.tokenizer)
            
            # get only generated tokens
            gen_tokens = [gt[prompt_padded_len:] for gt in gen_tokens]
            gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            responses.append(gen_text[0])
        return responses


            




    





