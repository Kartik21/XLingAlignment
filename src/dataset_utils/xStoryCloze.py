from datasets import load_dataset #load_dataset from Huggingface
from dataset_utils.abstractdataset import AbstractDataset 
from collections import defaultdict


'XStoryCloze: A New Large-Scale Cloze Dataset, Mostafazadeh et al., 2016'
'Task category : Common Sense Reasoning'


class XSTORYCLOZE(AbstractDataset):
    def __init__(self):
        super(XSTORYCLOZE, self).__init__()
        self.langkeys = {
                'english': 'en',
                'arabic': 'ar',
                'spanish': 'es',
                'basque': 'eu',
                'hindi': 'hi',
                'indonesian' : 'id',
                'burmese': 'my',
                'russian': 'ru',
                'swahili': 'sw',
                'telugu': 'te',
                'chinese': 'zh'}
    
    
    def get_dataset(self, args, lang):
   
        # Load the xstorycloze dataset
        dataset = load_dataset("juletxara/xstory_cloze", lang)

        # Get the dataset for the language
        dataset_sent1 = dataset[args.strata]['input_sentence_1']
        dataset_sent2 = dataset[args.strata]['input_sentence_2']
        dataset_sent3 = dataset[args.strata]['input_sentence_3']
        dataset_sent4 = dataset[args.strata]['input_sentence_4']
        dataset_ans1 = dataset[args.strata]['sentence_quiz1']
        dataset_ans2 = dataset[args.strata]['sentence_quiz2']
        answer_right_ending = dataset[args.strata]['answer_right_ending']

        assert len(dataset_sent1) == len(dataset_sent2) == len(dataset_sent3) == len(dataset_sent4) == len(dataset_ans1) == len(dataset_ans2)
    
        if args.combination_approach == "concatenation":

            '''
            if args.combination approach is concatenation, then premise, choice1 and choice2 are concatenated together for each 
            sentence in the dataset, thus keeping the number of datapoints the same. 
            '''
            data = []
            label = []
            idx = []
            for i in range(len(dataset_sent1)):
                data.append(dataset_sent1[i] + " " + dataset_sent2[i] + " " + dataset_sent3[i] + " " + dataset_sent4[i] + " " + dataset_ans1[i] + " " + dataset_ans2[i])
                label.append(answer_right_ending[i])
                idx.append(i)

        if args.combination_approach == "separate_inputs":
            '''
            if args.combination appraoch is separate_inputs, then sentence1, sentence2, sentence3, sentence4, quiz1, and quiz2 are separate samples 
            '''
            data = []
            label = []
            idx = []
            

            for i in range(len(dataset_sent1)):
                data.append(dataset_sent1[i])
                data.append(dataset_sent2[i])
                data.append(dataset_sent3[i])
                data.append(dataset_sent4[i])
                data.append(dataset_ans1[i])
                data.append(dataset_ans2[i])
                idx.append(i)
                label.append(answer_right_ending[i])
        return data, label, idx
    
    def get_dataset_accuracy(self, args, lang):
        # Load the xstorycloze dataset
        dataset = load_dataset("juletxara/xstory_cloze", lang)
        # Get the dataset for the language
        dataset_sent1 = dataset[args.strata]['input_sentence_1']
        dataset_sent2 = dataset[args.strata]['input_sentence_2']
        dataset_sent3 = dataset[args.strata]['input_sentence_3']
        dataset_sent4 = dataset[args.strata]['input_sentence_4']
        dataset_ans1 = dataset[args.strata]['sentence_quiz1']
        dataset_ans2 = dataset[args.strata]['sentence_quiz2']
        dataset_label = dataset[args.strata]['answer_right_ending']

        assert len(dataset_sent1) == len(dataset_sent2) == len(dataset_sent3) == len(dataset_sent4) == len(dataset_ans1) == len(dataset_ans2) == len(dataset_label)
        data = []
        answer = []
        prefix = "Based on the story and two choices, select the choice that is the most likely continuation of the story."
        for i in range(len(dataset_sent1)):
            prompt = prefix + "\nStory: " + dataset_sent1[i] + " " + dataset_sent2[i] + " " + dataset_sent3[i] + " " + dataset_sent4[i] + "\n0. " + dataset_ans1[i] + "\n1. " + dataset_ans2[i] + "\nThe correct choice number is: "
            data.append(prompt)
            answer.append(dataset_label[i]-1)
        return data, answer














    
