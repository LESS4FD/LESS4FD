import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
import torch.nn.init as init
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, auc



class LESS4FD(nn.Module):
    def __init__(self, dataset, args):
        super(LESS4FD, self).__init__()
        args.K = args.lg_k
        self.L_GPR = GPRGNN(dataset, args)
        args.K = args.gg_k
        self.G_GPR = GPRGNN(dataset, args)
        self.reset_parameters()

    def forward(self, data):
        
        h1_l, h2_l, prob_local = self.L_GPR(data)

        h1_g, h2_g, prob_global = self.G_GPR(data)

        return h1_l, h2_l, prob_local, h1_g, h2_g, prob_global

    def get_temp(self):
        return self.L_GPR.prop1.temp, self.G_GPR.prop1.temp
    
    def reset_parameters(self):
        for m in self.L_GPR.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
        for m in self.G_GPR.modules():
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, a=-0.1, b=0.1)  
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init='PPR', Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha]= 1.0
        elif self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3/(self.K+1))
            torch.nn.init.uniform_(self.temp,-bound,bound)
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden, dtype=torch.float64)
        self.lin2 = Linear(args.hidden, dataset.num_classes, dtype=torch.float64)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h1 = F.dropout(x, p=self.dropout, training=self.training)
        h1 = F.relu(self.lin1(h1))
        h2 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.lin2(h2)

        if self.dprate == 0.0:
            h2 = self.prop1(h2, edge_index)
            return h1, h2, F.log_softmax(h2, dim=1)
        else:
            h2 = F.dropout(h2, p=self.dprate, training=self.training)
            h2 = self.prop1(h2, edge_index)
            return h1, h2, F.log_softmax(h2, dim=1)
        
class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x

def train(model, data, optimizer, idx_train, idx_val, idx_test, labels, args, 
          fold, train_metrics, val_metrics, test_metrics):
    
    unsup_idx = torch.cat((idx_val, idx_test)).to(data.x.device)
    with tqdm(total=args.epochs, desc='(Training GCPR4FD)', disable=not args.verbose) as pbar:
        for epoch in range(1, 1+args.epochs):
            model.train()
            optimizer.zero_grad()
            
            loss = 0
            h1_l, h2_l, log_prob_local, h1_g, h2_g, log_prob_global = model(data)
            num_fake = data.y.sum().double()
            num_real = data.y.shape[0] - num_fake
            weights = [1.,num_real/num_fake]
            weights = torch.tensor(weights).double()

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            log_prob_local = log_prob_local.to(device)
            labels = labels.to(device)
            log_prob_global = log_prob_global.to(device)
            weights = weights.to(device)
            loss +=args.lambda_ce * (F.nll_loss(log_prob_local[idx_train], labels[idx_train], weight=weights) + args.lambda_g * F.nll_loss(log_prob_global[idx_train], labels[idx_train], weight=weights))
 
            if args.lambda_cr > 0:
                if args.onlyUnlabel == "yes":
                    loss_cr = consis_loss(args.cr_loss, [log_prob_local[unsup_idx], log_prob_global[unsup_idx]], args.cr_tem, args.cr_conf,args.lambda_g)
                else:
                    loss_cr = consis_loss(args.cr_loss, [log_prob_local, log_prob_global], args.cr_tem, args.cr_conf)
                loss += args.lambda_cr * loss_cr

            loss.backward()
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                model.eval()
                h1_l, h2_l, log_prob_local, h1_g, h2_g, log_prob_global = model(data)
                y_final = args.beta * log_prob_local + (1-args.beta) * log_prob_global
                auc_list1 = validation(y_final, labels, idx_train, train_metrics, fold)
                auc_list2 = validation(y_final, labels, idx_val, val_metrics, fold)
                auc_list3 = validation(y_final, labels, idx_test, test_metrics, fold)
                torch.save(auc_list3,f'./results/{args.dataset}_auc.pt')
            if epoch % args.eval_freq == 0 and args.verbose == True:
                print('Epoch {}, Train acc: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f} \n'.format(epoch, train_metrics.metrics['accs'][f'fold{fold+1}'][-1], val_metrics.metrics['accs'][f'fold{fold+1}'][-1], test_metrics.metrics['accs'][f'fold{fold+1}'][-1]))
                
                print('Epoch {}, Train precision: {:.4f}, Val precision: {:.4f}, Test precision: {:.4f} \n'.format(epoch, train_metrics.metrics['precisions'][f'fold{fold+1}'][-1], val_metrics.metrics['precisions'][f'fold{fold+1}'][-1], test_metrics.metrics['precisions'][f'fold{fold+1}'][-1]))
                
                print('Epoch {}, Train recall: {:.4f}, Val recall: {:.4f}, Test recall: {:.4f} \n'.format(epoch, train_metrics.metrics['recalls'][f'fold{fold+1}'][-1], val_metrics.metrics['recalls'][f'fold{fold+1}'][-1], test_metrics.metrics['recalls'][f'fold{fold+1}'][-1]))

                print('Epoch {}, Train f1: {:.4f}, Val f1: {:.4f}, Test f1: {:.4f} \n'.format(epoch, train_metrics.metrics['f1s'][f'fold{fold+1}'][-1], val_metrics.metrics['f1s'][f'fold{fold+1}'][-1], test_metrics.metrics['f1s'][f'fold{fold+1}'][-1]))
                
                print('Epoch {}, Train auc: {:.4f}, Val auc: {:.4f}, Test auc: {:.4f} \n'.format(epoch, train_metrics.metrics['aucs'][f'fold{fold+1}'][-1], val_metrics.metrics['aucs'][f'fold{fold+1}'][-1], test_metrics.metrics['aucs'][f'fold{fold+1}'][-1]))
                
                print('Epoch {}, Train apr: {:.4f}, Val apr: {:.4f}, Test apr: {:.4f} \n'.format(epoch, train_metrics.metrics['aprs'][f'fold{fold+1}'][-1], val_metrics.metrics['aprs'][f'fold{fold+1}'][-1], test_metrics.metrics['aprs'][f'fold{fold+1}'][-1]))

            pbar.set_postfix({'loss': loss.detach().cpu().item()})
            pbar.update()

    return loss.item()

def validation(y_final, labels, idx, metric, fold):

    y_pred = y_final.argmax(dim=-1, keepdim=True)[idx]
    labels = labels[idx]

    metric.metrics['accs'][f'fold{fold+1}'].append(accuracy_score(labels.cpu(), y_pred.cpu()))
    metric.metrics['precisions'][f'fold{fold+1}'].append(precision_score(labels.cpu(), y_pred.cpu(), average='macro'))
    metric.metrics['recalls'][f'fold{fold+1}'].append(recall_score(labels.cpu(), y_pred.cpu(), average='macro'))
    metric.metrics['f1s'][f'fold{fold+1}'].append(f1_score(labels.cpu(), y_pred.cpu(), average='macro'))
    metric.metrics['aucs'][f'fold{fold+1}'].append(roc_auc_score(labels.cpu(), y_pred.cpu(), average='macro'))
    y_test = y_final.cpu()[idx]
    fpr, tpr, thresholds = roc_curve(labels.cpu(), y_test.cpu()[:, 1])
    roc_auc = auc(fpr, tpr)
    auc_list = [fpr, tpr, thresholds, roc_auc]

    metric.metrics['aprs'][f'fold{fold+1}'].append(average_precision_score(labels.cpu(), y_pred.cpu(), average='macro'))
    return auc_list

def consis_loss(cr_loss, logps, tem, conf, lambda_g, w=1.0, reduction='none'):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    sum_p = ps[0] + lambda_g * ps[1]
    avg_p = sum_p/len(ps)
    avg_p = sum_p/len(ps)

    sharp_p = (torch.pow(avg_p, 1./tem) / torch.sum(torch.pow(avg_p, 1./tem), dim=1, keepdim=True)).detach()
    loss = 0.
    for i,p in enumerate(ps):
        if cr_loss == 'kl':
            log_p = torch.log(p)
            kl_loss = -F.kl_div(log_p, sharp_p, reduction='none').sum(1)
            if reduction != 'none':
                filtered_kl_loss = kl_loss[avg_p.max(1)[0] > conf]
                loss += torch.mean(filtered_kl_loss)
            else:
                loss += (w * kl_loss).mean()
                if i == 1:
                    loss = loss * lambda_g

        elif cr_loss == 'l2':
            if reduction != 'none':
                loss +=  torch.mean((p-sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf])
            else:
                loss += (w[avg_p.max(1)[0] > conf] * (p-sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf]).mean()
        else:
            raise ValueError(f"Unknown loss type: {cr_loss}")
    loss = loss/len(ps)
    return loss