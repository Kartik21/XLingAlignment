from comet import download_model, load_from_checkpoint

class COMET:
    def __init__(self, checkpoint = "Unbabel/wmt22-cometkiwi-da"):
        checkpoint_KIWI = "Unbabel/wmt22-cometkiwi-da"
        cometKIWI_path = download_model(checkpoint_KIWI)
        cometKIWI_model = load_from_checkpoint(cometKIWI_path)
        self.cometKIWI_model = cometKIWI_model

        checkpoint = "Unbabel/wmt22-comet-da"
        comet_path = download_model(checkpoint)
        comet_model = load_from_checkpoint(comet_path)
        self.comet_model = comet_model


    def run_COMETKIWI(self, consolidated_list):
        '''
        consolidated_list: 
        [{"src": "source text", 
        "mt": "machine translation", 
        "ref": "reference translation"},
        ...]
        '''
        
        #COMETKIWI requires the input to be in the following format:

        '''
        src_mt_list = [{"src": "source text",
                        "mt": "machine translation"}, 
                        ...]
        
        '''

        src_mt_list = [{key: value for key, value in d.items() if key!= "ref"} for d in consolidated_list]

        print("Running COMETKIWI evaluation on the generated outputs")
        
        self.cometKIWI_score = self.cometKIWI_model.predict(src_mt_list, batch_size=8, gpus=1)
        return self.cometKIWI_score
    
    def run_COMET(self, consolidated_list):
        '''
        consolidated_list: 
        [{"src": "source text",
        "mt": "machine translation",
        "ref": "reference translation"},
        ...]
        '''

        
        print("Running COMET evaluation on the generated outputs")
        
        self.comet_score = self.comet_model.predict(consolidated_list, batch_size=8, gpus=1)
        return self.comet_score



