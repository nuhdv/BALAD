import argparse
import numpy as np
import time
from src.evaluation.base import get_metrics
from src.evaluation.pa import pa_adjust_scores
from utils import get_data_lst, get_sub_seqs_label
from src.models.balad import BALAD
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=f'./datasets/')
parser.add_argument('--data', type=str,
                    default='sSWaT',
                    help='dataset name',
                    choices=['sASD', 'sSMD', 'sDSADS', 'sPSM', 'sMSL', 'sSWaT'])
parser.add_argument('--stride', help='', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--epoch_steps', type=int, default=40)
parser.add_argument('--rep_dim', help='', type=int, default=64)
parser.add_argument('--hidden_dims', help='', type=str, default='64')
parser.add_argument('--act', help='', type=str, default='ReLU')
parser.add_argument('--lr', help='', type=float, default=0.0003)
parser.add_argument('--batch_size', help='', type=int, default=128)
parser.add_argument('--bias', help='', type=bool, default=False)
parser.add_argument('--query_iters', help='', type=int, default=1)
parser.add_argument('--query_num', help='', type=int, default=5)
parser.add_argument('--runs', help='', type=int, default=1)
parser.add_argument('--split', help='', type=float, default=0.05)
parser.add_argument('--gamma', help='', type=float, default=1)
parser.add_argument('--lambd', help='', type=float, default=1)
parser.add_argument('--eta', help='', type=float, default=1)
parser.add_argument('--alpha', help='', type=float, default=0.1)

args = parser.parse_args()

model_configs = {
    'epochs': args.num_epochs,
    'epoch_steps': args.epoch_steps,
    'batch_size': args.batch_size,
    'lr': args.lr,
    'seq_len': args.seq_len,
    'stride': args.stride,
    'rep_dim': args.rep_dim,
    'hidden_dims': args.hidden_dims,
    'act': args.act,
    'query_iters': args.query_iters,
    'query_num': args.query_num,
    'gamma': args.gamma,
    'lambd': args.lambd,
    'eta': args.eta,
    'alpha': args.alpha
}

def eval(model, data, label):

    scores = model.decision_function(data, label)

    eval_metrics = get_metrics(label, pa_adjust_scores(label, scores))
    return eval_metrics

datasets = args.data.split(',')
for dataset in datasets:
    print(dataset)
    print("AUC-ROC,AUC-PR")
    eval_metrics_lst = []

    data, label, name = get_data_lst(args.data, args.data_root)
    train_data, train_label = data[:int(len(data) * args.split)], label[:int(len(data) * args.split)]
    test_data, test_label = data[int(len(data) * args.split):], label[int(len(data) * args.split):]

    start_time = time.time()
    for i in range(args.runs):
        print(f'\n\nRunning [{name}]  [{i+1}/{args.runs}], '
              f'cur_time: {time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())}')

        windows_train_label = get_sub_seqs_label(y=train_label, seq_len=args.seq_len, stride=args.stride)
        windows_test_label = get_sub_seqs_label(y=test_label, seq_len=args.seq_len, stride=args.stride)

        model = BALAD(**model_configs, random_state=42+i)
        model.fit(train_data, np.zeros_like(windows_train_label))
        torch.save(model.net.state_dict(), 'sSWaT_pretrain_ml.pt')
        eval_m = eval(model, test_data, windows_test_label)
        txt = f'{name}, '
        txt += ', '.join(['%.4f' % a for a in eval_m])
        txt += f', runs {i + 1}/{args.runs}'
        print(txt)
        eval_metrics_lst.append(eval_m)

    avg, std = np.average(np.array(eval_metrics_lst), axis=0), np.std(np.array(eval_metrics_lst), axis=0)

    txt = f'{name}, '
    txt += ', '.join(['%.4f' % a for a in avg])
    txt += f', avg\n'
    txt += f'{name}, '
    txt += ', '.join(['%.4f' % a for a in std])
    txt += f', std'
    print(txt)