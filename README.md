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
To validate the efficacy of our approach, we conducted comprehensive experiments, detailed as follows (we refer to the implementation using GPT-3.5 as $LESS4FD$* and the implementation with Llama2 as $LESS4FD^{\diamond}$) :

![Overall Results_1](/figs/res_all_1.png "Overall results w.r.t accuracy and F1 score") <br>
*Overall results w.r.t accuracy and F1 score* <br> <br>

![Overall Results_2](/figs/res_all_2.png "Overall results w.r.t precision and recall") <br>
*Overall results w.r.t accuracy and F1 score*  <br> <br>

![aucs](/figs/aucs.png "ROC curves") <br>
*ROC curves*  <br> <br>

![Abaltion Results_1](/figs/ablation_1.png "Abaltion results of $LESS4FD*$") <br>
*Abaltion results of LESS4FD<sup>*</sup>*  <br> <br>

![Abaltion Results_2](/figs/ablation_1.png "Abaltion results of $LESS4FD^{\diamond}$") <br>
*Abaltion results of __LESS4FD<sup>⋄</sup>__*  <br> <br>

![Computational Costs](/figs/cost.png "Computational Costs") <br>
*Computational Costs*  <br> <br>

![Topics_1](/figs/topics_gpt.png "Performance on different topics of $LESS4FD$*") <br>
*Performance on different topics of <strong>LESS4FD<sup>*</sup></strong>*  <br> <br>

![Topics_2](/figs/topics_llama2.png "Performance on different topics of  $LESS4FD^{\diamond}$") <br>
*Performance on different topics of __LESS4FD<sup>⋄</sup>__*  <br> <br>
