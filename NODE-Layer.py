import numpy as np
import pandas as pd
import torchdiffeq
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
from torchcubicspline import(natural_cubic_spline_coeffs,
                             NaturalCubicSpline)

EF_BIN_NUM = 3
EF_BINS = None

class _ODEFunc(nn.Module):
    def __init__(self):
        super(_ODEFunc, self).__init__()
        self.autonomous = True
        self.weights = torch.nn.Parameter(
            torch.FloatTensor(10))
        self.weights.data = torch.linspace(0, 1, 10)
        y_data = torch.ones_like(self.weights.view(-1)).to(device) * (-1) ** (torch.arange(len(self.weights.view(-1))).to(device))
        self.y_data = y_data.unsqueeze(-1)

    def generate_ode_function(self,):
        try:
            coeffs = natural_cubic_spline_coeffs(self.weights, self.y_data)
            spline_function = NaturalCubicSpline(coeffs)
            # spline_function = CubicSpline(spline_x_data.detach().cpu().numpy(), self.y_data, bc_type=((2, 0), (2, 0)))
        except Exception as e:
            spline_function = None
            print(e)
        return spline_function

    def generate_func(self):
        def cal(x):
            return x * self.weights[0] * (x - self.weights[1]) * (x - self.weights[2])
        return cal

    def forward(self, t, x):
        if not self.autonomous:
            x = torch.cat([torch.ones_like(x[:, [0]]) * t, x], 1)
            return x
        else:
            spline_function = self.generate_ode_function()
            # spline_function = self.generate_func()
            # result = x * self.weights[0] * (x - self.weights[1]) * (x - self.weights[2])
            # return spline_function(x)
            # return torch.tensor(spline_function(x.view(-1)).reshape(x.shape), requires_grad= True, dtype=torch.float32).to(device)
            return spline_function.evaluate(x.view(-1))
            # x_detach = x.view(-1).detach().cpu().numpy()
            # return torch.tensor(spline_function(x_detach).reshape(x.shape), requires_grad= True, dtype=torch.float32).to(device)

class NODEQuantizer(nn.Module):
    def __init__(self, solver: str = 'dopri5', rtol: float = 1e-4, atol: float = 1e-4, adjoint: bool = False
                 ):
        super().__init__()
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.use_adjoint = adjoint
        # self.integration_time = torch.tensor([0, 1], dtype=torch.float32)
        self.integration_time = torch.linspace(0, 1, 10)
        # self.ode_function = _ODEFunc([]).to(device)
        self.ode_funcs = nn.ModuleList()
        for i, (col, bins) in enumerate(EF_BINS.items()):
            ode_function = _ODEFunc().to(device)
            self.ode_funcs.append(ode_function)

    def init_ode_functions(self):
        ode_functions = []
        for i, (col, bins) in enumerate(EF_BINS.items()):
            ode_function = _ODEFunc().to(device)
            ode_functions.append(ode_function)
        return ode_functions

    def get_ef_bin(self, x, levels=EF_BIN_NUM):
        percentile_centroids = torch.quantile(x, torch.linspace(0, 100, levels + 1)[1:-1].to(device)/ 100, dim=0).T
        return percentile_centroids.view(-1)

    def forward(self,  x: torch.Tensor, adjoint: bool = True, integration_time=None):
        integration_time = self.integration_time if integration_time is None else integration_time
        integration_time = integration_time.to(x.device)
        if len(x.size()) !=1:
            input_dimensions = torch.split(x, 1, dim=1)
        else:
            input_dimensions = torch.split(x, 1, dim=0)
        ode_method = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        try:
            # ode_outputs = [ode_method(odc_fun, feature, integration_time, rtol=self.rtol,  atol=self.atol, method=self.solver)[-1] for odc_fun, feature in zip(self.ode_funcs, input_dimensions)]
            ode_outputs = []
            for odc_fun, feature in zip(self.ode_funcs, input_dimensions):
               out = ode_method(odc_fun, feature, integration_time, rtol=self.rtol, atol=self.atol, method='rk4')[-1]
               ode_outputs.append(torch.clip(out, 0, 1))
            if len(x.size()) != 1:
                combined_output = torch.cat(ode_outputs, dim=1)
            else:
                combined_output = torch.cat(ode_outputs, dim=0)
            return combined_output
        except Exception as e:
            print(e)
            # ode_outputs = []
            # for odc_fun, feature in zip(self.ode_funcs, input_dimensions):
            #     out = ode_method(odc_fun, feature, integration_time, rtol=self.rtol, atol=self.atol, method='rk4')[
            #         -1]
            #     ode_outputs.append(out)
            # combined_output = torch.cat(ode_outputs, dim=1)
            # return combined_output
            # ef_outputs = [torch.bucketize(feature, self.get_ef_bin(feature)) for feature in input_dimensions]
            # combined_ef_output = torch.cat(ef_outputs, dim=1)
            # return torch.tensor(combined_ef_output, dtype=torch.float32, requires_grad=True)

class GermanNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(GermanNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.node_layer = NODEQuantizer()

    def forward(self, x):
        e = self.node_layer(x)
        h1 = self.relu(self.linear1(e))
        h2 = self.relu(self.linear2(h1))
        h3 = self.relu(self.linear2(h2))
        h4 = self.relu(self.linear2(h3))
        h5 = self.relu(self.linear2(h4))
        h6 = self.relu(self.linear2(h5))
        a6 = self.linear3(h6)
        y = self.softmax(a6)
        return y

batch_size = 100

def get_weights(df, target, show_heatmap=False):
    def heatmap(cor):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()
    cor = df.corr()
    cor_target = abs(cor[target])
    weights = cor_target[:-1]  # removing target WARNING ASSUMES TARGET IS LAST
    weights = weights / np.linalg.norm(weights)
    if show_heatmap:
        heatmap(cor)
    return weights.values

from Atttacks import lowProFool, deepfool, fgsm

#import Adverse as lpf_multiclass
def gen_adv(model, X_test, y_test, method,alpha, lambda_,bounds, maxiters=10, weights=None, bar=True):
    results = np.zeros_like(X_test)
    iter = range(X_test.shape[0])
    for i in iter:
        x = X_test[i]
        y = y_test[i]
        x_tensor = torch.FloatTensor([x]).to(device)
        try:
            if method == 'LowProFool':
                orig_pred, adv_pred, x_adv, loop_i = lowProFool(x=x_tensor, model=model, weights=weights, bounds=bounds,
                                                                 maxiters=maxiters, alpha=alpha, lambda_=lambda_, device=device)
            elif method == 'Deepfool':
                orig_pred, adv_pred, x_adv, loop_i = deepfool(x_tensor, model, maxiters, alpha,
                                                           bounds, weights=[], device=device)
            elif method == 'FGSM':
                x_adv = fgsm(x_tensor, model, eps=alpha, device=device)
            else:
                raise Exception("Invalid method", method)
            results[i] = x_adv
        except:
            print(f"generate adversarial example for the {i}-th data instance failed.")
    return results

def get_ef_bins(x, levels = EF_BIN_NUM):
    ef_bins = {}
    columns = ['column'+ str(i) for i in range(x.shape[1])]
    percentile_centroids = np.quantile(x, np.linspace(0, 100, levels + 1)[1:-1]/100, axis=0).T
    for i, col in enumerate(columns):
        ef_bins[col] = percentile_centroids[i]
    return ef_bins

def train_model(X_train, y_train, es=True, es_count=10):
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.eye(2)[y_train.astype(int)].to(device)
    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    D_in = X_train.size(1)
    D_out = y_train.size(1)
    epochs = 100
    H = 100

    model = GermanNet(D_in, H, D_out).to(device)
    model.train()

    lr = 1e-4
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    cur_es_count = 0
    min_loss = 999999
    # Training in batches
    for epoch in range(epochs):
        current_loss = 0
        current_correct = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            output = model(inputs)
            _, indices = torch.max(output, 1)  # argmax of output [[0.61,0.12]] -> [0]
            preds = torch.eye(2)[indices].to(device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            current_correct += (preds.int() == labels.int()).sum() / D_out
        current_loss = current_loss / len(torch_dataset)
        current_correct = current_correct / len(torch_dataset)
        if es:
            if min_loss > current_loss:
                min_loss = current_loss
                cur_es_count = 0
            else:
                cur_es_count += 1
                if cur_es_count >= es_count:
                    print("> epoch {:.0f}\tLoss {:.5f}\tAcc {:.5f}".format(epoch, current_loss, current_correct))
                    print("early stopped.")
                    return model
        if (epoch % 25 == 0):
            print("> epoch {:.0f}\tLoss {:.5f}\tAcc {:.5f}".format(epoch, current_loss, current_correct))
    return model

def torch_predict(model, data):
    data_tensor = torch.FloatTensor(data).to(device)
    model.eval()
    y_pred = model(data_tensor).max(1)[1]
    return y_pred

def evaluate_data(model, y_true, data, pure_pred=None, log=True):
    data_tensor = torch.FloatTensor(data).to(device)
    y_pred = model(data_tensor).max(1)[1]
    acc = accuracy_score(y_true, y_pred.detach().cpu().numpy())
    #f1 = f1_score(y_true, y_pred.detach().cpu().numpy())
    auc = roc_auc_score(y_true, y_pred.detach().cpu().numpy())
    print(f"acc: {acc}, auc: {auc}")
    return acc, auc
    # if pure_pred is not None:
    #     suc_rate = torch.sum(y_pred != pure_pred)/len(data)
    #     if log:
    #         print(f"acc: {acc}, success rate: {suc_rate}")
    #     return acc, suc_rate
    # else:
    #     return acc

dataset_list = [
    # 'Covtype',
    # 'SkinSegmentation',
    # 'magic',
    # 'waveform-5000',
    # 'spambase',
    # 'wall-following',
    # 'pendigits',
    # 'Occupancy',
    'segment',
]

SEED = 0

#attack param: 攻击参数
maxiters = 50
eps = 0.1
lambda__ = 10 # for LowProFool

from pathlib import Path
out_path = Path('./D3-ODE/')

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

for dataset_name in dataset_list:
    print(dataset_name)
    df = pd.read_csv(f'./uci-datasets/float/{dataset_name}.csv')
    df.fillna(0, inplace=True)

    if dataset_name in ['Covtype', 'Higgs', 'SkinSegmentation', 'accelerometer']:
        sample_df = pd.concat([df[df.iloc[:,-1] == 0].sample(5000, random_state=SEED),
                               df[df.iloc[:,-1] == 1].sample(5000, random_state=SEED)])
        df = sample_df

    X = df.iloc[:,:-1].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = df.iloc[:,-1].values
    y = LabelEncoder().fit_transform(y)
    if dataset_name == 'satellite':
        idx = np.bitwise_or(y == 0, y == 3)
        X = X[idx]
        y = y[idx]
        y[y == 0] = 0
        y[y == 3] = 1
    else:
        X = X[y < 2]
        y = y[y < 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=len(X) // 2, shuffle=True, random_state=SEED, stratify=y)

    EF_BINS = get_ef_bins(X_train, EF_BIN_NUM)
    _d3_path = out_path / f'{dataset_name}_d3.pt'
    if _d3_path.exists():
        model = torch.load(_d3_path)
    else:
        model = train_model(X_train, y_train)
        torch.save(model, _d3_path)
    model.eval()
    y_pred = model(torch.FloatTensor(X_test).to(device))
    y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
    acc0 = accuracy_score(y_test, y_pred)
    auc0 = roc_auc_score(y_test, y_pred)
    print("Accuracy score on test data", acc0)
    print("AUC0 on test data", auc0)
    print("Train D3 Layer finished !!!")

    weights = get_weights(df, df.columns[-1])
    bounds = X.min(axis=0), X.max(axis=0)

    if len(X_test) > 300:
        np.random.seed(SEED)
        random_permute = np.random.permutation(len(X_test))
        X_test_sample = X_test[random_permute[:300]]
        y_test_sample = y_test[random_permute[:300]]
    else:
        X_test_sample = X_test.to(device)
        y_test_sample = y_test.to(device)
    #
    # Generate adversarial examples
    adv_samples = {}
    for attack_name in ['FGSM', 'LowProFool', 'Deepfool']:
        par_list = []
        if len(par_list) > 0:
            for par in par_list:
                _p = out_path / f'{dataset_name}_{par}_{attack_name.lower()}.npy'
                if _p.exists():
                    adv_samples[attack_name + par ] = np.load(_p)
        else:
            _p = out_path / f'{dataset_name}_{attack_name.lower()}.npy'
            if _p.exists():
                adv_samples[attack_name] = np.load(_p)
            else:
                print(X_test_sample.shape)
                x_adv = gen_adv(model, X_test_sample, y_test_sample, attack_name, eps, lambda__, bounds, maxiters, weights)
                np.save(_p, x_adv)
                adv_samples[attack_name] = x_adv
    #
    acc1, auc1 = evaluate_data(model, y_test_sample, X_test_sample)

    print("Accuracy score on 300 test data", acc1)
    print("AUC on 300 test data", auc1)

    with open(out_path / f'./clean_result_d3Layer.csv', mode='a') as f:
        f.write(','.join([
            dataset_name,
            str(acc1.item()),
            str(auc1.item()),
        ]))
        f.write('\n')

    for atk_name, atk_raw in adv_samples.items():
        acc_a, auc_a = evaluate_data(model, y_test_sample, atk_raw)
        with open(out_path/f'./adv_attack_d3Layer.csv', mode='a') as f:
            f.write(','.join([
                dataset_name,
                atk_name,
                str(acc_a.item()),
                str(auc_a.item())
            ]))
            f.write('\n')

