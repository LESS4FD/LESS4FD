import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from model.LESS4FD import LESS4FD, train
from arguments import arg_parser
from metrics import My_metrics
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import os
from utils import load_data

def run_ours():
    args = arg_parser()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args)
    split_dir = f"./splits/{args.dataset}"
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        
    skf = StratifiedShuffleSplit(n_splits = 10, test_size=1 - args.tr)
                    
    train_indices, val_indices, test_indices = [], [], []
    
    for i, spt in enumerate(skf.split(np.zeros(data.n_news), data.y)):
        train_index = spt[0]
        test_index = spt[1]
        train_indices.append(train_index)
        val_indices.append(test_index[:int(len(test_index)/2)])
        test_indices.append(test_index[int(len(test_index)/2):])

        np.savetxt(f"{split_dir}/train_fold{i}.txt", train_index)
        np.savetxt(f"{split_dir}/val_fold{i}.txt", test_index[:int(len(test_index)/2)])
        np.savetxt(f"{split_dir}/test_fold{i}.txt", test_index[int(len(test_index)/2):])

    train_metrics = My_metrics(folds=args.fold)
    val_metrics = My_metrics(folds=args.fold)
    test_metrics = My_metrics(folds=args.fold)

    for fold in range(args.fold):

        model = LESS4FD(data, args)
        model = model.to(device)

        data = data.to(device)
        
        labels = data.y
        idx_train, idx_val, idx_test = train_indices[fold], val_indices[fold], test_indices[fold]

        idx_train = torch.from_numpy(idx_train)
        idx_val = torch.from_numpy(idx_val)
        idx_test = torch.from_numpy(idx_test)

        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
        
        loss = train(model, data, optimizer, idx_train, idx_val, idx_test, labels, args, fold, train_metrics, val_metrics, test_metrics)
        print('-'* 60)
        
        print('[FOLD {} BEST] val acc: {:.4f}, test acc: {:.4f} \n'.format(fold, val_metrics.get_fold_best('accs', f'fold{fold+1}')[0], test_metrics.get_item('accs', f'fold{fold+1}',val_metrics.get_fold_best('accs', f'fold{fold+1}')[1])))
 
        print('[FOLD {} BEST] val precision: {:.4f}, test precision: {:.4f} \n'.format(fold, val_metrics.get_fold_best('precisions', f'fold{fold+1}')[0], test_metrics.get_item('precisions', f'fold{fold+1}', val_metrics.get_fold_best('precisions', f'fold{fold+1}')[1])))
        
        print('[FOLD {} BEST] val recall: {:.4f}, test recall: {:.4f} \n'.format(fold, val_metrics.get_fold_best('recalls', f'fold{fold+1}')[0], test_metrics.get_item('recalls', f'fold{fold+1}', val_metrics.get_fold_best('recalls', f'fold{fold+1}')[1])))
        
        print('[FOLD {} BEST] val f1: {:.4f}, test f1: {:.4f} \n'.format(fold, val_metrics.get_fold_best('f1s', f'fold{fold+1}')[0], test_metrics.get_item('f1s', f'fold{fold+1}', val_metrics.get_fold_best('f1s', f'fold{fold+1}')[1])))
        
        print('[FOLD {} BEST] val auc: {:.4f}, test auc: {:.4f} \n'.format(fold, val_metrics.get_fold_best('aucs', f'fold{fold+1}')[0], test_metrics.get_item('aucs', f'fold{fold+1}', val_metrics.get_fold_best('aucs', f'fold{fold+1}')[1])))
        
        print('[FOLD {} BEST] val apr: {:.4f}, test apr: {:.4f} \n'.format(fold, val_metrics.get_fold_best('aprs', f'fold{fold+1}')[0], test_metrics.get_item('aprs', f'fold{fold+1}', val_metrics.get_fold_best('aprs', f'fold{fold+1}')[1])))

    facc = '[FINAL ACC] {:.4f}+-{:.4f} \n'.format(test_metrics.get_final('accs')[0], test_metrics.get_final('accs')[1])
    fpre = '[FINAL PRECISION] {:.4f}+-{:.4f} \n'.format(test_metrics.get_final('precisions')[0], test_metrics.get_final('precisions')[1])
    frec = '[FINAL RECALL] {:.4f}+-{:.4f} \n'.format(test_metrics.get_final('recalls')[0], test_metrics.get_final('recalls')[1])
    ff1 = '[FINAL F1] {:.4f}+-{:.4f} \n'.format(test_metrics.get_final('f1s')[0], test_metrics.get_final('f1s')[1])
    fauc = '[FINAL AUC] {:.4f}+-{:.4f} \n'.format(test_metrics.get_final('aucs')[0], test_metrics.get_final('aucs')[1])
    fapr = '[FINAL APR] {:.4f}+-{:.4f} \n'.format(test_metrics.get_final('aprs')[0], test_metrics.get_final('aprs')[1])

    print(facc, fpre, frec, ff1, fauc, fapr)
    
    with open(f'./results/LEG4FD_{args.dataset}_{args.num_topics}_{args.topic_type}.txt', 'a') as file:
        file.write(f"LEG4FD 10-fold test on {args.dataset}_epochs_{args.epochs}_cr_{args.lambda_cr}_lgk_{args.lg_k}_ggk_{args.gg_k}_ce_{args.lambda_ce}: {facc}\n")
        file.write(f"LEG4FD 10-fold test on {args.dataset}_epochs_{args.epochs}_cr_{args.lambda_cr}_lgk_{args.lg_k}_ggk_{args.gg_k}_ce_{args.lambda_ce}: {fpre}\n")
        file.write(f"LEG4FD 10-fold test on {args.dataset}_epochs_{args.epochs}_cr_{args.lambda_cr}_lgk_{args.lg_k}_ggk_{args.gg_k}_ce_{args.lambda_ce}: {frec}\n")
        file.write(f"LEG4FD 10-fold test on {args.dataset}_epochs_{args.epochs}_cr_{args.lambda_cr}_lgk_{args.lg_k}_ggk_{args.gg_k}_ce_{args.lambda_ce}: {ff1}\n")
        file.write(f"LEG4FD 10-fold test on {args.dataset}_epochs_{args.epochs}_cr_{args.lambda_cr}_lgk_{args.lg_k}_ggk_{args.gg_k}_ce_{args.lambda_ce}: {fauc}\n")
        file.write(f"LEG4FD 10-fold test on {args.dataset}_epochs_{args.epochs}_cr_{args.lambda_cr}_lgk_{args.lg_k}_ggk_{args.gg_k}_ce_{args.lambda_ce}: {fapr}\n")

if __name__ == '__main__':
    run_ours()
