import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData, Data
import xlsxwriter as xw
from deepwalk import graph
import argparse


def load_edge(dataset,num_topics,node):
    news_index = np.load(f'../Data/{dataset}/graph/nodes/news_index.npy',allow_pickle= True).item()
    if node == 'topic':
        df = pd.read_excel(f'../Data/{dataset}/graph/edges/news2topic_{num_topics}.xlsx')
        index_dict = np.load(f'../Data/{dataset}/graph/nodes/{node}_index_{num_topics}.npy',allow_pickle= True).item()
    else:
        df = pd.read_excel(f'../Data/{dataset}/graph/edges/news2{node}.xlsx')
        index_dict = np.load(f'../Data/{dataset}/graph/nodes/{node}_index.npy',allow_pickle= True).item()
    pair = df.values.tolist()
    edges = []
    edges_ = []
    for i in pair:
        head = news_index[i[0]]
        tail = index_dict[str(i[1])]
        edge = [head, tail]
        edge_ = [tail, head]
        edges.append(edge)
        edges_.append(edge_) 
    return edges,edges_

def build_hg(dataset,num_topics):
    news_attr = np.load(f'../Data/{dataset}/Embeddings/news_embeddings.npy')
    news_attr = torch.from_numpy(news_attr)
    entity_attr = np.load(f'../Data/{dataset}/Embeddings/entity_embeddings.npy')
    entity_attr = torch.from_numpy(entity_attr)
    topic_attr = np.load(f'../Data/{dataset}/Embeddings/topic_embeddings_{num_topics}.npy')    
    topic_attr = torch.from_numpy(topic_attr)

    news2entity, news2entity_ = load_edge(dataset,num_topics,'entity')
    news2topic, news2topic_ = load_edge(dataset,num_topics,'topic')
    df_news = pd.read_excel(f'../Data/{dataset}/news_final.xlsx')
    label = df_news['label'].tolist()
    
    data = HeteroData()
    data['news'].x = news_attr
    data['entity'].x = entity_attr
    data['topic'].x = topic_attr
    data['news', 'has', 'entity'].edge_index = torch.tensor(news2entity, dtype=torch.long).t().contiguous()
    data['entity', 'has_1', 'news'].edge_index = torch.tensor(news2entity_, dtype=torch.long).t().contiguous()
    data['news', 'belongs', 'topic'].edge_index = torch.tensor(news2topic, dtype=torch.long).t().contiguous()
    data['topic', 'belongs_1', 'news'].edge_index = torch.tensor(news2topic_, dtype=torch.long).t().contiguous()
    data['news'].y = torch.tensor(label,dtype = torch.long)
    print('='*60)
    print('HeteroGraph:',dataset,'\n',data)
    print(' num_nodes:',data.num_nodes,'\n','num_edges:',data.num_edges,'\n','Data has isolated nodes:',data.has_isolated_nodes(),'\n','Data is undirected:',data.is_undirected())
    print('='*60,'\n')
    torch.save(data,f'../Data/{dataset}/graph/{dataset}_{num_topics}.pt')
    return data

def class2global(edgelist,global_index,classindex):
    indices_g = []
    for i in edgelist:
        ID = classindex[i]
        index_g = global_index[ID]
        indices_g.append(index_g)
    return indices_g

def get_edgeList(dataset,num_topics):
    news_index = np.load(f'../Data/{dataset}/graph/nodes/news_index.npy', allow_pickle=True).item()
    entity_index = np.load(f'../Data/{dataset}/graph/nodes/entity_index.npy', allow_pickle=True).item()
    topic_index = np.load(f'../Data/{dataset}/graph/nodes/topic_index_{num_topics}.npy', allow_pickle=True).item()
    data = torch.load(f'../Data/{dataset}/graph/{dataset}_{num_topics}.pt')
    del data['entity', 'has_1', 'news']
    del data['topic', 'belongs_1', 'news']

    newsList0 = data['news', 'has', 'entity'].edge_index.tolist()[0]
    entityList = data['news', 'has', 'entity'].edge_index.tolist()[1]
    newsList1 = data['news', 'belongs', 'topic'].edge_index.tolist()[0]
    topicList = data['news', 'belongs', 'topic'].edge_index.tolist()[1]
    global_index = np.load(f'../Data/{dataset}/graph/nodes/global_index_{num_topics}.npy', allow_pickle=True).item()

    news0_g = class2global(newsList0,global_index,news_index)
    entity_g = class2global(entityList,global_index,entity_index)

    news1_g = class2global(newsList1,global_index,news_index)
    topic_g = class2global(topicList,global_index,topic_index)   

    node_head = news0_g + news1_g   
    node_tail = entity_g + topic_g   

    edgeList = []
    edgeList_rw = []
    for i in range(len(node_head)):
        head = node_head[i]
        tail = node_tail[i]
        edge = [head,tail]
        edge_rw = str(head)+' '+str(tail)
        edgeList.append(edge)
        edgeList_rw.append(edge_rw)
    with open(f'../Data/{dataset}/graph/edges/{dataset}_{num_topics}.edgelist','w',encoding = 'utf-8') as f:
        for i in edgeList_rw:
            f.write(str(i)+'\n')
        f.close()

    data_rw = Data(edge_index=torch.tensor(edgeList, dtype=torch.long).t().contiguous())
    G_rw = graph.load_edgelist(f'../Data/{dataset}/graph/edges/{dataset}_{num_topics}.edgelist',undirected = True)
    assert G_rw.number_of_nodes() == data_rw.num_nodes and G_rw.number_of_edges() == data_rw.num_edges,'wrong graph'
    return edgeList_rw



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='choose dataset')
    parser.add_argument('--dataset', type=str, default='MM COVID', help="['MM COVID','ReCOVery','MC Fake']")
    parser.add_argument('--num_topics', type=int)
    args = parser.parse_args()
    dataset = args.dataset
    num_topics = args.num_topics
    
    graph = build_hg(dataset,num_topics)       
    edgeList_rw = get_edgeList(dataset,num_topics)
    
    print(f'graph & edgelist for {dataset} done') 
