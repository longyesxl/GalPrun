import vgg
import torch
import math
from optimizer import Optimizer, required

class FISTA(Optimizer):

    def __init__(self, params, lr=0.01, Lamb=0.05 , lr_scaled=0.1,lr_scaled_time=30):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if Lamb < 0.0:
            raise ValueError("Invalid Lamb value: {}".format(Lamb))
        self.ak=1
        self.ak1=1
        self.epoch=1
        defaults = dict(lr=lr,  Lamb=Lamb,lr_scaled=lr_scaled,lr_scaled_time=lr_scaled_time)
        super(FISTA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FISTA, self).__setstate__(state)

    def step1(self, closure=None):
        self.ak1=0.5*(1+math.sqrt(1+4*self.ak*self.ak))
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            if(self.epoch%group['lr_scaled_time']==0):
                group['lr']*=group['lr_scaled']
            for p in group['params'] :

                param_state = self.state[p]
                if 'm_buffer' not in param_state:
                    param_state['m_buffer'] = torch.zeros_like(p.data)
                if 'y_buffer' not in param_state:
                    param_state['y_buffer'] = torch.zeros_like(p.data)
                
                m_buf = param_state['m_buffer']
                y_buf = param_state['y_buffer']

                y_buf = p.data+(self.ak-1)/self.ak1*(p.data-m_buf)
                m_buf = p.data
                p.data = y_buf
        self.ak=self.ak1
        return loss
    def step2(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                y_now = param_state['y_buffer']
                rz=y_now-group['lr']*d_p
                p.data = torch.sign(rz)*torch.nn.functional.relu(torch.abs(rz)-group['lr']*group['Lamb'])
        self.epoch+=1
        return loss

