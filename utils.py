import torch
from torch_geometric.data import HeteroData, Data

def load_data(args):
    
    graph = torch.load(f'../HeteroGraphs/{args.dataset}_{args.num_topics}.pt')
  
    news_x = graph['news'].x
    labels = graph['news'].y
    entity_x = graph['entity'].x
    topic_x = graph['topic'].x

    x = torch.cat((news_x, entity_x), dim=0)
    x = torch.cat((x, topic_x), dim=0)

    graph['news','has','entity'].edge_index[1] = graph['news','has','entity'].edge_index[1] + graph['news'].num_nodes
    graph['entity', 'has_1', 'news'].edge_index[0] = graph['entity', 'has_1', 'news'].edge_index[0] + graph['news'].num_nodes

    graph['news','belongs','topic'].edge_index[1] = graph['news','belongs','topic'].edge_index[1] + graph['news'].num_nodes + graph['entity'].num_nodes
    graph['topic', 'belongs_1', 'news'].edge_index[0] = graph['topic', 'belongs_1', 'news'].edge_index[0] + graph['news'].num_nodes + graph['entity'].num_nodes

    homo_edge_list = torch.cat((graph['news','has','entity'].edge_index, graph['news','belongs','topic'].edge_index ), dim=1)
    homo_edge_list = torch.cat((homo_edge_list, graph['topic', 'belongs_1', 'news'].edge_index), dim=1)
    homo_edge_list = torch.cat((homo_edge_list, graph['entity', 'has_1', 'news'].edge_index), dim=1)
    homo_edge_list = homo_edge_list.to(torch.long)

    homoGraph = Data(
        x =x,
        edge_index=homo_edge_list,
        y=labels,
        des='news, entity, topic',
        n_news = graph['news'].num_nodes,
        num_classes = 2
    )

    return homoGraph
