import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
#from torch.utils.data import TensorDataset, DataLoader
from DataGenModule import p1db_12 as wavefunction

# Configuration
m = 50 # Number of measurements
n = 50 # Number of time insants
layers = [32,64,8,64,32]
activation = nn.Tanh
lr, lr1, lr2 = 1e-4, 1e-3,1e-3
# a,b = 1, 1 # for lbfgs
# a for w, b for f
a,b = 1,0.01 # for adam
split_ratio = 0.7
num_epochs = int(1e6)
adam = True
weight_import = True

def initcuda():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)
    return device

def generate_data(n,m,f=wavefunction):
    x_val = np.linspace(0, 1, m)
    t_val = np.linspace(0, 2, n)
    X, T = np.meshgrid(x_val, t_val)

    psi_matrix = f(X, T)
    return X, T, psi_matrix

def train_create(full=False):
    X, T, psi_matrix = generate_data(n,m)
    X_array = X.flatten().astype(np.float32)
    T_array = T.flatten().astype(np.float32)
    W_array = psi_matrix.flatten()

    split_index = int(split_ratio * X_array.size)

    # train_input_x = np.concatenate((X_array[:int(split_index/2)], X_array[split_index:]))
    # train_input_t = np.concatenate((T_array[:int(split_index/2)], T_array[split_index:]))
    # train_target  = np.concatenate((W_array[:int(split_index/2)], W_array[split_index:]))

    train_input_x = X_array[:int(split_index)]
    train_input_t = T_array[:int(split_index)]
    train_target  = W_array[:int(split_index)]

    # test_input_x = X_array[split_index-5:split_index]
    # test_input_t = T_array[split_index-5:split_index]
    # test_target = W_array[split_index-5:split_index]
    
    if full:
        ret_arr = [X_array, T_array, W_array, X_array, T_array, W_array, layers, activation, a,b] 
    else:
        ret_arr = [train_input_x, train_input_t, train_target, X_array, T_array, W_array, layers, activation, a,b]
    with open('array_param', 'wb') as fp:
        pickle.dump(ret_arr, fp)

    return ret_arr

def train_model(model):
    if weight_import == True:
        model.dnn.load_state_dict(torch.load('./modelparam.pth'))
        print('loaded previous weights')
    if type(model.optimizer) == torch.optim.LBFGS:
        model.train()

    else:
        for i in tqdm(range(num_epochs)):
            model.train()

class DNN(nn.Module):
    """return Vanilla Fully Connencted Neural Network"""
    

    def __init__(self, hidden_sizes, activation_fn):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(2, hidden_sizes[0]))
        self.layers.append(activation_fn())
        # nn.init.xavier_normal_(self.layers[0].weight)
        # nn.init.zeros_(self.layers[0].bias)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            """use array of numbers to input hidden layers with numbers
            representing neurons in the corresponding hidden layers"""
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(activation_fn())
            # nn.init.xavier_normal_(self.layers[i+1].weight)
            # nn.init.zeros_(self.layers[i+1].bias)
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], 2))
        self.layers.apply(init_weights)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[:,0], x[:,1]

# the physics-guided neural network
class ComplexNN():
    def __init__(self, X_array, T_array, W_array, X_full, T_full, W_full, hidden_sizes, activation_fn, a, b):
        device = initcuda()       
        # data
        self.x = torch.tensor(X_array[:], requires_grad=True).float().to(device)
        self.xf= torch.tensor(X_full[:], requires_grad=True).float().to(device)
        self.t = torch.tensor(T_array[:], requires_grad=True).float().to(device)
        self.tf= torch.tensor(T_full[:], requires_grad=True).float().to(device)
        self.w = torch.tensor(W_array[:], requires_grad=True).float().to(device)
        self.train_losses = []
        self.train_losses_w = []
        self.train_losses_f = []
        self.a = a
        self.b = b
        
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn
        
        # deep neural networks
        self.dnn = DNN(hidden_sizes, activation_fn).to(device)
        
        # optimizers: using the same settings
        if adam:
            self.optimizer = torch.optim.AdamW(
                self.dnn.parameters(),
                lr=lr)
        else:
            self.optimizer = torch.optim.LBFGS(
                self.dnn.parameters(), 
                lr=1, 
                max_iter=5000, 
                max_eval=10000, 
                history_size=200,
                tolerance_grad=1e-9, 
                tolerance_change=1.0 * np.finfo(float).eps,
                line_search_fn="strong_wolfe"       # can be "strong_wolfe"
            )
        # self.optimizer1 = torch.optim.Adam(
        #     self.dnn.parameters(),
        #     lr=lr1)
        # self.optimizer2 = torch.optim.Adam(
        #     self.dnn.parameters(),
        #     lr=lr2)
        self.iter = 0

        # scheduler
        # self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
    def net_w(self, x, t):  
        R, I = self.dnn(torch.stack([x, t], dim=1))
        return R**2 + I**2
    
    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        R, I = self.dnn(torch.stack([x, t], dim=1))

        R_t = torch.autograd.grad(
            R, t, 
            grad_outputs=torch.ones_like(R),
            retain_graph=True,
            create_graph=True
        )[0]
        I_t = torch.autograd.grad(
            I, t, 
            grad_outputs=torch.ones_like(I),
            retain_graph=True,
            create_graph=True
        )[0]        
        R_x = torch.autograd.grad(
            R, x, 
            grad_outputs=torch.ones_like(R),
            retain_graph=True,
            create_graph=True
        )[0]
        R_xx = torch.autograd.grad(
            R_x, x, 
            grad_outputs=torch.ones_like(R_x),
            retain_graph=True,
            create_graph=True
        )[0]
        I_x = torch.autograd.grad(
            I, x, 
            grad_outputs=torch.ones_like(I),
            retain_graph=True,
            create_graph=True
        )[0]
        I_xx = torch.autograd.grad(
            I_x, x, 
            grad_outputs=torch.ones_like(I_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        f = R*(R_t + I_xx/2) + I*(I_t - R_xx/2)
        return f 
    
    def net_f2(self, x, t):
        """output is already squared"""
        R, I = self.dnn(torch.stack([x, t], dim=1))

        R_t = torch.autograd.grad(
            R, t, 
            grad_outputs=torch.ones_like(R),
            retain_graph=True,
            create_graph=True
        )[0]
        I_t = torch.autograd.grad(
            I, t, 
            grad_outputs=torch.ones_like(I),
            retain_graph=True,
            create_graph=True
        )[0]        
        R_x = torch.autograd.grad(
            R, x, 
            grad_outputs=torch.ones_like(R),
            retain_graph=True,
            create_graph=True
        )[0]
        R_xx = torch.autograd.grad(
            R_x, x, 
            grad_outputs=torch.ones_like(R_x),
            retain_graph=True,
            create_graph=True
        )[0]
        I_x = torch.autograd.grad(
            I, x, 
            grad_outputs=torch.ones_like(I),
            retain_graph=True,
            create_graph=True
        )[0]
        I_xx = torch.autograd.grad(
            I_x, x, 
            grad_outputs=torch.ones_like(I_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = (0.5*R_xx - I_t)**2 + (0.5*I_xx + R_t)**2
        return f
    
    def loss_func(self):
        self.optimizer.zero_grad()
        
        w_pred = self.net_w(self.x, self.t)
        f_pred = self.net_f2(self.xf, self.tf)
        loss_w = torch.mean((self.w - w_pred) ** 2)
        loss_f = torch.mean(f_pred)
        
        loss = self.a*loss_w + self.b*loss_f
        
        loss.backward()
        self.iter += 1
        self.train_losses.append(loss.item())
        self.train_losses_w.append(loss_w.item())
        self.train_losses_f.append(loss_f.item())

        if self.iter % 100 == 0:
            # print(
            #     'Epoch %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            # )
            torch.save(self.dnn.state_dict(), "./modelparam.pth")
            if len(self.train_losses) > 1000:
                with open('loss_list', 'wb') as fp:
                    pickle.dump([self.train_losses[-1000:],self.train_losses_f[-1000:],self.train_losses_w[-1000:]], fp)

        return loss
    
    def loss_func1(self):
        self.optimizer.zero_grad()
        
        w_pred = self.net_w(self.x, self.t)
        loss = a*torch.mean((self.w - w_pred) ** 2)
        loss.backward()
        self.iter += 1
        # if self.iter % 100 == 0:
        #     print(
        #         'Epoch %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
        #     )
        self.train_losses_w.append(loss.item())
        return loss

    def loss_func2(self):
        self.optimizer.zero_grad()
        
        f_pred = self.net_f(self.x, self.t)
        loss = b*torch.mean(f_pred ** 2)
        
        loss.backward()
        # self.iter += 1
        # if self.iter % 100 == 0:
        #     print(
        #         'Epoch %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
        #     )
        self.train_losses_f.append(loss.item())
        return loss

    
    def train(self):
        self.dnn.train()
                
        # Backward and optimize
        self.optimizer.step(self.loss_func)
        # lf = self.loss_func()
        # self.scheduler.step(self.loss_func())
        # self.optimizer1.step(self.loss_func1)
        # self.optimizer2.step(self.loss_func2)


            
    def predict(self, X_array, T_array):
        x = torch.tensor(X_array, requires_grad=True).float().to(device)
        t = torch.tensor(T_array, requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_w(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

if __name__ == '__main__':
    model = ComplexNN(*train_create(full=False))
    train_model(model)