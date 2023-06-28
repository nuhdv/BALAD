from torch.utils.data import DataLoader, TensorDataset
from src.models.core.base_model import BaseDeepAD
from .core.strategy import *
from src.models.mlp import MLPnet
from src.models.one_class import BALADLoss
import numpy as np

torch.backends.cudnn.enabled = False # to avoid cudnn_status_internal_error

class BALAD(BaseDeepAD):
    def __init__(
            self,
            # training parameters
            epochs=10, batch_size=128, lr=3e-4, seq_len=100, stride=1, epoch_steps=40,
            # network architecture parameters
            rep_dim=64, hidden_dims='64', act='ReLU', bias=False,
            # other parameters
            prt_steps=1, device='cuda', verbose=2, query_iters=1, query_num=5, gamma=1, lambd=1, eta=1, alpha=0.1, random_state=42
    ):
        super(BALAD, self).__init__(
            model_name='BALAD', epochs=epochs, batch_size=batch_size, lr=lr, seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias
        self.query_iters = query_iters
        self.query_num = query_num
        self.gamma = gamma
        self.lambd = lambd
        self.eta = eta
        self.alpha = alpha

        self.c = None
        return

    def training_prepare(self, X, y):

        dataset = TensorDataset(torch.from_numpy(X).float(),
                                torch.from_numpy(y).long())

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        net = NetworkModule(
            n_features=self.n_features,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation=self.act,
            bias=False
        ).to(self.device)

        self.c = self._set_c(net, train_loader)
        criterion = BALADLoss(self.c, nu=self.alpha)

        return train_loader, net, criterion

    def training_forward(self, batch_x, net, criterion, up_R):
        batch_x, batch_y = batch_x
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        rep = net(batch_x)
        loss = criterion(rep, batch_y, up_R)
        return loss

    def inference_prepare(self, X, label):

        dataset = TensorDataset(torch.from_numpy(X).float(),
                                torch.from_numpy(label).long())
        test_loader = DataLoader(dataset, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        self.criterion.reduction = 'mean'
        return test_loader

    def inference_forward(self, batch_x, batch_label, net, criterion, optimizer):

        batch_x = batch_x.float().to(self.device)
        if len(batch_x) != self.batch_size:
            rep = net(batch_x)
            score = torch.sum((rep - self.c) ** 2, dim=1)
            return rep,score
        rep_lst = []
        score_lst = []
        query_indexs_lst = np.array([])
        query_pool = -np.ones(batch_label.shape[0])
        for i in range(self.query_iters):
            rep = net(batch_x)
            score = torch.sum((rep - self.c) ** 2, dim=1) - criterion.R ** 2
            s_p = score.sigmoid()
            compact_R = torch.sum((torch.index_select(rep, dim=0, index=torch.where(score<0)[0]) - self.c) ** 2, dim=1)
            compact_R = torch.sqrt(torch.mean(compact_R))  #compact_R is nan means all the points is outliers
            relaxation_R = torch.sum((torch.index_select(rep, dim=0, index=torch.where(score>0)[0]) - self.c) ** 2, dim=1)
            relaxation_R = torch.sqrt(torch.mean(relaxation_R)) # relaxation_R is nan means all the points is inliers
            dist = torch.sqrt(torch.sum(torch.abs(rep - self.c) ** 2, dim=1))

            lower_bound = (1 - self.alpha) * criterion.R + self.alpha * (criterion.R if torch.isnan(compact_R) else compact_R)
            up_bound = (1 - self.alpha) * criterion.R + self.alpha * (criterion.R if torch.isnan(relaxation_R) else relaxation_R)

            query_indexs = self.query(self.query_num, query_pool, dist=dist, lower_bound=lower_bound, R=criterion.R, up_bound=up_bound)
            query_pool[query_indexs] = batch_label[query_indexs]
            query_indexs_lst = np.append(query_indexs_lst, query_indexs)

            loss_unlabel = self.sigmoid_entropy(score[np.where(query_pool == -1)[0]])  # select the unlabeled sample to minimize entropy
            s_p_label = torch.cat((1 - s_p[np.where(query_pool == 0)[0]], s_p[np.where(query_pool == 1)[0]]))
            loss_entropy = self.focal_entropy(s_p_label)
            loss_penalize = criterion(rep, torch.tensor(query_pool).long().to(self.device), True)
            loss_label = loss_penalize + self.lambd * loss_entropy
            loss = loss_label + self.eta * loss_unlabel

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            rep_lst.append(rep)
            score_lst.append(score)

        return rep_lst[0], score_lst[0]

    def focal_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Based on focal loss, Entropy of binary distribution from logits."""
        return -((1 - x) ** self.gamma * torch.log(x)).mean(0)

    def sigmoid_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of binary distribution from logits."""
        return -(torch.nn.Sigmoid()(x) * torch.nn.LogSigmoid()(x)).mean(0)
    def _set_c(self, net, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        net.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                out = net(x)
                if type(out) is tuple:
                    z = out[0]
                else:
                    z = out
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)

        # if c is too close to zero, set to +- eps
        # a zero unit can be trivially matched with zero weights
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def query(self, query_num, label, dist=None, lower_bound=None, R=None, up_bound=None):
        unlabel_indexs = np.where(label == -1)[0]
        query_select = Boundary_skip_Sampling(unlabel_indexs, dist, lower_bound, R, up_bound)
        query_indexs = query_select.query(query_num)

        return query_indexs

class NetworkModule(torch.nn.Module):
    def __init__(
            self, n_features, hidden_dims='64', rep_dim=64, activation='ReLU', bias=False):
        super(NetworkModule, self).__init__()

        network_params = {
            'n_features': n_features,
            'n_hidden': hidden_dims,
            'n_output': rep_dim,
            'activation': activation,
            'bias': bias
        }

        self.neural_network = MLPnet(**network_params)
        return

    def forward(self, x):
        rep = self.neural_network(x)
        return rep