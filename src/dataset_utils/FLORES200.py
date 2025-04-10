from datasets import load_dataset #load_dataset from Huggingface
from dataset_utils.abstractdataset import AbstractDataset 
from collections import defaultdict


class FLORES200(AbstractDataset):
    def __init__(self):
        #Inherited from AbstractDataset class
        super(AbstractDataset, self).__init__()
        self.langkeys =  {'mesopotamian arabic': 'acm_Arab',
                         'afrikaans': 'afr_Latn',
                         'tosk albanian': 'als_Latn',
                         'amharic': 'amh_Ethi',
                            'north levantine arabic': 'apc_Arab',
                            'modern standard arabic': 'arb_Arab',
                            'modern standard arabic_romanized': 'arb_Latn',
                            'najdi arabic': 'ars_Arab',
                            'moroccan arabic': 'ary_Arab',
                            'egyptian arabic': 'arz_Arab',
                            'assamese': 'asm_Beng',
                            'north azerbaijani': 'azj_Latn',
                            'bambara': 'bam_Latn',
                            'bengali': 'ben_Beng',
                            'bengali romanized': 'ben_Latn',
                            'standard tibetan': 'bod_Tibt',
                            'bulgarian': 'bul_Cyrl',
                            'catalan': 'cat_Latn',
                            'cebuano': 'ceb_Latn',
                            'czech': 'ces_Latn',
                            'central kurdish': 'ckb_Arab',
                            'danish': 'dan_Latn',
                            'german': 'deu_Latn',
                            'greek': 'ell_Grek',
                            'english': 'eng_Latn',
                            'estonian': 'est_Latn',
                            'basque': 'eus_Latn',
                            'finnish': 'fin_Latn',
                            'french': 'fra_Latn',
                            'nigerian fulfulde': 'fuv_Latn',
                            'west central oromo': 'gaz_Latn',
                            'guarani': 'grn_Latn',
                            'gujarati': 'guj_Gujr',
                            'haitian creole': 'hat_Latn',
                            'hausa': 'hau_Latn',
                            'hebrew': 'heb_Hebr',
                            'hindi': 'hin_Deva',
                            'hindi romanized': 'hin_Latn',
                            'croatian': 'hrv_Latn',
                            'hungarian': 'hun_Latn',
                            'armenian': 'hye_Armn',
                            'igbo': 'ibo_Latn',
                            'ilocano': 'ilo_Latn',
                            'indonesian': 'ind_Latn',
                            'icelandic': 'isl_Latn',
                            'italian': 'ita_Latn',
                            'javanese': 'jav_Latn',
                            'japanese': 'jpn_Jpan',
                            'jingpho': 'kac_Latn',
                            'kannada': 'kan_Knda',
                            'georgian': 'kat_Geor',
                            'kazakh': 'kaz_Cyrl',
                            'kabuverdianu': 'kea_Latn',
                            'halh mongolian': 'khk_Cyrl',
                            'khmer': 'khm_Khmr',
                            'kinyarwanda': 'kin_Latn',
                            'kyrgyz': 'kir_Cyrl',
                            'korean': 'kor_Hang',
                            'lao': 'lao_Laoo',
                            'lingala': 'lin_Latn',
                            'lithuanian': 'lit_Latn',
                            'ganda': 'lug_Latn',
                            'luo': 'luo_Latn',
                            'standard latvian': 'lvs_Latn',
                            'malayalam': 'mal_Mlym',
                            'marathi': 'mar_Deva',
                            'macedonian': 'mkd_Cyrl',
                            'maltese': 'mlt_Latn',
                            'maori': 'mri_Latn',
                            'burmese': 'mya_Mymr',
                            'dutch': 'nld_Latn',
                            'norwegian bokmal': 'nob_Latn',
                            'nepali': 'npi_Deva',
                            'nepali romanized': 'npi_Latn',
                            'northern sotho': 'nso_Latn',
                            'nyanja': 'nya_Latn',
                            'odia': 'ory_Orya',
                            'eastern panjabi': 'pan_Guru',
                            'southern pashto': 'pbt_Arab',
                            'western persian': 'pes_Arab',
                            'plateau malagasy': 'plt_Latn',
                            'polish': 'pol_Latn',
                            'portuguese': 'por_Latn',
                            'romanian': 'ron_Latn',
                            'russian': 'rus_Cyrl',
                            'shan': 'shn_Mymr',
                            'sinhala romanized': 'sin_Latn',
                            'sinhala': 'sin_Sinh',
                            'slovak': 'slk_Latn',
                            'slovenian': 'slv_Latn',
                            'shona': 'sna_Latn',
                            'sindhi': 'snd_Arab',
                            'somali': 'som_Latn',
                            'southern sotho': 'sot_Latn',
                            'spanish': 'spa_Latn',
                            'serbian': 'srp_Cyrl',
                            'swati': 'ssw_Latn',
                            'sundanese': 'sun_Latn',
                            'swedish': 'swe_Latn',
                            'swahili': 'swh_Latn',
                            'tamil': 'tam_Taml',
                            'telugu': 'tel_Telu',
                            'tajik': 'tgk_Cyrl',
                            'tagalog': 'tgl_Latn',
                            'thai': 'tha_Thai',
                            'tigrinya': 'tir_Ethi',
                            'tswana': 'tsn_Latn',
                            'tsonga': 'tso_Latn',
                            'turkish': 'tur_Latn',
                            'ukrainian': 'ukr_Cyrl',
                            'urdu': 'urd_Arab',
                            'urdu romanized': 'urd_Latn',
                            'northern uzbek': 'uzn_Latn',
                            'vietnamese': 'vie_Latn',
                            'waray': 'war_Latn',
                            'wolof': 'wol_Latn',
                            'xhosa': 'xho_Latn',
                            'yoruba': 'yor_Latn',
                            'simplified chinese': 'zho_Hans',
                            'traditional chinese': 'zho_Hant',
                            'standard malay': 'zsm_Latn',
                            'zulu': 'zul_Latn'}      
    def get_dataset(self, args, lang):
        #Load the dataset from Huggingface
        dataset = load_dataset("facebook/flores", 'all', trust_remote_code=True)
        dev_dataset = dataset['dev']
        test_dataset = dataset['devtest']
        data = []
        if args.strata == 'test':
            for i in range(len(test_dataset)):
                data.append(test_dataset[i]['sentence_'+ lang])
        else:
            for i in range(len(dev_dataset)):
                data.append(dev_dataset[i]['sentence_'+ lang])
        return data
                

        

        
       
        
                
        







