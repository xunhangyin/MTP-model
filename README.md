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

## Pruning

### File describe

ESM_trainer.py : the file to define and train the model

get_data_new.py: The file to get the data from dataset

get_parse.py: the file to define the hyper-parameter

GO_process.py  // obo_proccess.py:The file to process GO-basic.obo to get the ancestor of each GO term

main.py :the main file

predict.py: the file to make use of the model to predict

ESM_model.py:the file to define ESM_LSTM model to pruning

### Directory Describe

Can_anc_files: the directory to store the file of GO_ancestor and the Candidate GO term

dataset:the directory to store the dataset

output_model : the directory to store the model finetuned

### Running The file

To begin with, the file to record the GO ancestor,the file to record the GO_id utilized in the Meta-learning is needed to be stored in the Can_anc_files directory.

Then ,

```
python main.py
```

to train the ESM_LSTM model to pruning.

The finetuned model is stored in the output_model directory

Lastly,

```
python predict.py
```

to get the prediction.

## Attention

**the file path should be edited if you run the code.**

**the dataset  needs to process firstly, the python file only read the file processed**

## Requirements

python==3.8

pytorch==2.1.0

transformers==4.40.0

chromadb

PIL

pandas