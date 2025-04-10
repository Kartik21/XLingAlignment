# Discriminative Alignment Index 

## Folder structure
Since not all folders are relevant to this project (and will be refactored shortly),  I have highlighted the pertinent files/folders below. 
### src/
- embed_extractor.py: Gets the embeddings based on the last token and the weighted embedding for each sample in a file. Outputs the embedding as a .pkl file
- compute_alignment_xstorycloze.py : Calculates the DALI for each xstorycloze sample based on the embeddings file
- compute_alignment_xcopa.py: Calculates the DALI for each xcopa sample based on the embeddings file
- compute_alignment_belebele.py: Calculates the DALI for each belebele sample based on the embeddings file

### data
Has the necessary .txt files that are used to extract embeddings
- belebele_dali/ : Has .txt files associated with 82 languages and 4 options for each
- belebele_mexa/: Has .txt files associated with 82 languages
- flores_200_dataset/devtest: Has ~1000 sentences for each language in the flores dataset (not a .txt file)
- xcopa_dali/: Has 500 sentences (2 options) for 10 languages including the corresponding English translation
- xstorycloze_dali: Has 1511 sentences (2 options) for 10 languages + English

### alignment_outputs/
- Has the necessary DALI files (as .json) by layer for each sample
- Each dali file is structured to be {Layer_id: {sample_id: 1/0}}

### src/plots/
- Has the xstorycloze and xcopa plotting file
- Can be combined into a single notebook later


### TBD



