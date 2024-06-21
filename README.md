# On Fake News Detection with LLM Enhanced Semantics Mining

### This repository is the official implementation of "On Fake News Detection with LLM Enhanced Semantics Mining" 

## Datasets

To access the datasets used in this study, please use the following links:

MM COVID: https://github.com/bigheiniu/MM-COVID

ReCOVery: http://coronavirus-fakenews.com

MC Fake: https://github.com/qwerfdsaplking/MC-Fake

PAN2020: https://pan.webis.de/data.html

LIAR: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

## To Run

Build heterogeneous graphs for each dataset:

```
python build_graph.py --dataset <dataset_name> --num_topics <num_of_topics>
```

Run HeteroSGT for fake news detection

```
python main.py --dataset <dataset_name> --num_topics <num_of_topics>
```

Run case study for $\lambda_{ce}$

```python
bash lambda_test.sh
```

## Experimental Results:
To validate the efficacy of our approach, we conducted comprehensive experiments, detailed as follows:

![Overall Results](/figs/res_all_1.png "Overall results w.r.t accuracy and F1 score ")
*Caption: Overall results w.r.t accuracy and F1 score*

