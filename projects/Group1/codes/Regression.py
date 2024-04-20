import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
N = 3           # Number of eigenstates to consider 
k = 0.01        # Value of hbar^2/2m
V = 0           # Potential is zero or not


def initcuda():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)
    return device

def sinf(x,N):
    '''Function used inside psi'''
    return torch.sin( torch.tensor(list(i*x*torch.pi for i in range(1,N+1)),dtype=torch.float32) )

def psi2_zero(X,T,a,phi):
    '''Takes an array of position and time, created a grid of PDF ( |psi|^2 ) using parameter tensors a and phi'''
    N = len(a)
    k = 1
    E = torch.tensor(list( k* (i**2) for i in range(1,N+1) )).to(device)
    ones1 = torch.ones(N).to(device)

    phi_m = torch.einsum('m,n->nm',phi,ones1).to(device)
    phi_n = torch.einsum('n,m->nm',phi,ones1).to(device)

    E_m = torch.einsum('m,n->nm',E,ones1).to(device)
    E_n = torch.einsum('n,m->nm',E,ones1).to(device)

    F = []
    for t in T:
        f = []
        for x in X:
            costerm = torch.cos(phi_m - phi_n - t*(E_m-E_n)).to(device)
            sinfterm = sinf(x,N).to(device)
            M = torch.einsum('m,n,m,n,nm->nm',a,a,sinfterm,sinfterm,costerm)
            f.append( torch.sum(M) ) 
        F.append(torch.stack(f))
    return torch.stack(F)


def psi2_nonzero(X,T,a,phi,b,theta,E):
    '''Takes an array of x i.e X and outputs an array of y: PYTORCH VERSION'''
    N = len(a)
    k = 0.01
    # E = torch.tensor(list( k* (i**2) for i in range(1,N+1) )).to(device)
    ones2 = torch.ones(N,N).to(device)
    ones3 = torch.ones(N,N,N).to(device)

    theta_mn = torch.einsum('mn,pq->mnpq',theta,ones2).to(device)
    theta_pq = torch.einsum('pq,mn->mnpq',theta,ones2).to(device)
    phi_m = torch.einsum('m,npq->mnpq',phi,ones3).to(device)
    phi_p = torch.einsum('p,mnq->mnpq',phi,ones3).to(device)
    E_m = torch.einsum('m,npq->mnpq',E,ones3).to(device)
    E_p = torch.einsum('p,mnq->mnpq',E,ones3).to(device)

    # a_m = torch.einsum('m,npq->mnpq',a,ones3)
    # a_p = torch.einsum('p,mnq->mnpq',a,ones3)
    # b_mn = torch.einsum('mn,pq->mnpq',b,ones2)
    # b_pq = torch.einsum('pq,mn->mnpq',b,ones2)

    F = []
    for t in T:
        f = []
        for x in X:
            costerm = torch.cos(phi_m - phi_p - t*(E_m-E_p) + theta_mn - theta_pq)
            sinfterm = sinf(x,N).to(device)
            M = torch.einsum('m,p,mn,pq,mnpq,n,q->mnpq',a,a,b,b,costerm,sinfterm,sinfterm)
            f.append( torch.sum(M) )
        F.append(torch.stack(f))
    return torch.stack(F)


def train_run(lossfn,optimizer,closure,optimizer2):
    ''' Default optimisation sequence for both V=0 and V!=0'''

    # LBFGS Initial Run
    num_epochs = int(5)
    losses = []
    for i in tqdm(range(num_epochs)):
        loss = lossfn()
        optimizer.step(closure)
        losses.append(loss.item())
    print(losses[-1])

    # LBFGS Final Run
    while losses[-2]-losses[-1]>0:
        loss = lossfn()
        if len(losses)%10==0:
            print(losses[-1])
        optimizer.step(closure)
        losses.append(loss.item())
    print(losses[-1])
    plt.plot(losses)
    plt.show()

    # Adam Initial Run
    losses2 = []
    num_epochs2 = 50
    for i in tqdm(range(num_epochs2)):
        loss = lossfn()
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        losses2.append(loss.item())
    plt.plot(losses2)
    plt.show()


    # Adam Final run
    while losses2[-1]>1e-9 or (losses2[-1]==losses2[-2] and losses2[-3]==losses2[-3]):
        loss = lossfn()
        if len(losses2)%50==0:
            print(losses2[-1])
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        losses2.append(loss.item())
    print(losses2[-1])
    plt.plot(losses2)
    plt.show()


def run_V_zero():
    '''Module to run the whole setup for V=0'''

    ones1 = torch.ones(N).to(device)

    a_true = torch.tensor([1/2**0.5,1/4**0.5, 1/4**0.5],dtype=torch.float32)
    phi_true = torch.tensor(np.random.rand(N),dtype=torch.float32)

    a = torch.tensor(np.random.rand(N),dtype=torch.float32,requires_grad=True)
    phi = torch.tensor(np.random.rand(N),dtype=torch.float32,requires_grad=True)

    xarr = np.linspace(0,1,25)
    tarr = np.linspace(0,3,5)

    result = psi2_zero(xarr,tarr,a_true,phi_true)

    lossfnmse = torch.nn.MSELoss()
    reg_param = 0
    def lossfn():
        mse = lossfnmse(psi2_zero(xarr,tarr,a,phi),result)
        reg_loss = torch.sum(torch.abs(a))
        return mse + reg_param*reg_loss
    
    optimizer = torch.optim.LBFGS((a,phi),
                    history_size=10, 
                    max_iter=4, 
                    line_search_fn="strong_wolfe")
    optimizer2 = torch.optim.AdamW((a,phi),lr=1e-5)

    def closure():
        optimizer.zero_grad()
        loss = lossfn()
        loss.backward()
        return loss
    
    train_run(lossfn,optimizer,closure,optimizer2)

    # Show results
    plt.plot(xarr,psi2_zero(xarr,[0.9],a,phi).detach().numpy().flatten(), label='psi2')
    plt.plot(xarr,psi2_zero(xarr,[0.9],a_true,phi_true).detach().numpy().flatten(), label='psi2_true')
    plt.legend()
    plt.show()

def run_V_nonzero():
    '''Module to run the whole setup for V=0'''

    a_true = torch.tensor(np.random.rand(N),dtype=torch.float32)
    a_true = a_true.div(torch.linalg.norm(a_true)).to(device)
    phi_true = torch.tensor(np.random.rand(N),dtype=torch.float32).to(device)
    b_true = torch.eye(N,dtype=torch.float32).to(device)
    theta_true = torch.zeros([N,N],dtype = torch.float32).to(device)
    E_true = torch.tensor(list( k* (i**2) for i in range(1,N+1) )).to(device)

    pre_a = torch.tensor(np.random.rand(N),dtype=torch.float32)
    a = pre_a.div(torch.linalg.norm(pre_a)).to(device).requires_grad_()
    b0 = torch.rand(N,N).to(device)
    b = b0 / torch.linalg.norm(b0, axis = 1, keepdims = True)
    phi = torch.tensor(np.random.rand(N),dtype=torch.float32).to(device).requires_grad_()
    E = torch.tensor(list( k* (i**2) for i in range(1,N+1) )).to(device)
    theta = torch.tensor(np.random.rand(N,N),dtype=torch.float32).to(device).requires_grad_()

    xarr = torch.linspace(0,1,20).to(device)
    tarr = torch.linspace(0,500,20).to(device)
    result = psi2_nonzero(xarr,tarr,a_true,phi_true, b_true, theta_true,E_true)

    lossfn = torch.nn.MSELoss()
    optimizer = torch.optim.LBFGS((a,phi,b,theta,E),
                        history_size=10, 
                        max_iter=4, 
                        line_search_fn="strong_wolfe")
    optimizer2 = torch.optim.AdamW((a,phi,b,theta,E),lr=1e-5)

    def closure():
        optimizer.zero_grad()
        loss = lossfn(psi2_nonzero(xarr,tarr,a,phi,b,theta,E),result) 
        loss.backward()
        return loss

    def lossf():
        return lossfn(psi2_nonzero(xarr,tarr,a,phi,b,theta,E),result)

    train_run(lossf,optimizer,closure,optimizer2)

    # Show results
    plt.plot(xarr,psi2_nonzero(xarr,[0.9],a,phi,b,theta,E).detach().numpy().flatten(), label='psi2')
    plt.plot(xarr,psi2_nonzero(xarr,[0.9],a_true,phi_true, b_true, theta_true,E_true).detach().numpy().flatten(), label='psi2_true')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    device = initcuda()
    if V==0:
        run_V_zero()

    elif V!=0:
        run_V_nonzero()