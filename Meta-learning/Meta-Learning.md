## Meta-Learning

### File-describe

data_process.py:    process the dataset to get the Leaf GO terms,three files:id-anno,anno-id,GO-ancestor are the output

ESM_trainer.py : the file to define and train the model

get_data_new.py: The file to get the data from dataset

get_parse.py: the file to define the hyper-parameter

GO_process.py  // obo_proccess.py:The file to process GO-basic.obo to get the ancestor of each GO term

main.py :the main file

ESM_predictor.py: prediction process to get the candidate GO terms

predict.py: the file to process ESM_predictor

### Directory Describe

ESM : storing the initial ESM weight

Chroma_file : to store the chroma data storing the embedding of each GO term

data_processed :  store the processed data

data : store the initial dataset

model_output: to store the trained model weight

### Attention

**the file path should be edited if you run the code.**

### Running The file

To begin with 

`python data_process.py`

to get the processed dataset to get the leaf GO_terms of each sample The result is stored in dataset_processed directory.

Then,

`python main.py`

to train the model

Lastly,

`python predict.py` 

to get the candidate GO terms.

