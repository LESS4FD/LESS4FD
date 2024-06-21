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
(We refer to the implementation using GPT-3.5 as $LESS4FD$* and the implementation with Llama2 as $LESS4FD^{\diamond}$.)

![Overall Results_1](/figs/res_all_1.png "Overall results w.r.t accuracy and F1 score")
*Overall results w.r.t accuracy and F1 score* <br>

![Overall Results_2](/figs/res_all_2.png "Overall results w.r.t precision and recall")
*Overall results w.r.t accuracy and F1 score*

![aucs](/figs/aucs.png "ROC curves")
*ROC curves*

![Abaltion Results_1](/figs/ablation_1.png "Abaltion results of $LESS4FD*$")
*Abaltion results of LESS4FD with GPT-3.5 derived embeddings*

![Abaltion Results_2](/figs/ablation_1.png "Abaltion results of $LESS4FD^{\diamond}$")
*Abaltion results of LESS4FD with Llama2 derived embeddings*

![Computational Costs](/figs/costs.png "Computational Costs")
*Computational Costs*

![Topics_1](/figs/topics_gpt.png "Performance on different topics of $LESS4FD$*")
*Performance on different topics of LESS4FD with GPT-3.5 derived embeddings*

![Topics_2](/figs/topics_llama2.png "Performance on different topics of  $LESS4FD^{\diamond}$")
*Performance on different topics of LESS4FD with Llama2 derived embeddings*
