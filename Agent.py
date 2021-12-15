import numpy as np
import torch
import torch.nn as nn

def fclayer(inchannels,outchannels):
    fc = nn.Sequential(nn.Linear(inchannels,outchannels),
                       nn.LeakyReLU())
    return fc

class Agent(nn.Module):
    def __init__(self,inshape = [7,7],hiddensize = 100,numlayers = 2,numreps = 5):
        super(Agent,self).__init__()
        #inputs will be small, so I'll flatten it and use a fully connected network
        self.insize = np.prod(inshape)
        self.hiddensize = hiddensize
        self.numlayers = numlayers
        self.numreps = numreps
        self.numparams = 0
        
        self.lstm = nn.LSTMCell(self.insize,self.hiddensize)
        self.hidden_state = (torch.zeros(1,self.hiddensize),
                             torch.zeros(1,self.hiddensize))
        
        self.outactions = nn.Linear(self.hiddensize,5) #5 movement actions, l,r,u,d,stay
        self.outreps = nn.Linear(self.hiddensize,self.numreps)
        
    def forward(self,x):
        out = x.reshape(x.size(0),-1) 
        self.hidden_state = self.lstm(out,self.hidden_state) 
        actions = self.outactions(self.hidden_state[0])
        reps = self.outreps(self.hidden_state[0])
        
        actions = nn.Softmax(dim = 1)(actions)
        actions = actions.flatten()
        
        if self.numreps != 1:
            reps = nn.Softmax(dim = 1)(reps)
            reps = reps.flatten()
                        
        return actions,reps
    
    def reset(self):
        self.hidden_state = (torch.zeros(1,self.hiddensize),
                             torch.zeros(1,self.hiddensize))
    
    def mutate(self,coef,mut):
        if coef != 0:
            for i,name in enumerate(self.state_dict()):
                torch.manual_seed(mut+i) #seed+i for each layer is still sampling from N,
                                          #it's just easier to do it for each layer individually
                shape = self.state_dict()[name].shape
                self.state_dict()[name] += coef * torch.empty(shape).normal_(mean=0,std=1)

#         else:
#             for j in range(len(coefs)):
#                 coef,mut = coefs[j],mutations[j]
#                 if coef != 0:
#                     for i,name in enumerate(self.state_dict()):
#                         torch.manual_seed(mut+i) #seed+i for each layer is still sampling from N,
#                                                   #it's just easier to do it for each layer individually
#                         shape = self.state_dict()[name].shape
#                         self.state_dict()[name] += coef * torch.empty(shape).normal_(mean=0,std=1)
        
    
    def save(self,filename,optimizer,measures = None):
        state = {
            'measures': measures,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filename)

    def load(self,filename, optimizer, measures = None):
        checkpoint = torch.load(filename, map_location = 'cpu')
        measures = checkpoint['measures']
        self.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return measures, optimizer
    
    def paramcount(self):
        if self.numparams == 0:
            for comp in self.state_dict():
                self.numparams += np.prod(agent.state_dict()[comp].shape)
        return self.numparams

def processobs(obs):
    obs = torch.Tensor(obs)
#     for i in range(len(obs)):
#         obs[i] = (obs[i] - obs[i].mean())/torch.sqrt(obs[i].var())
    return obs.unsqueeze(0)
    
def hammingdist(vecs,vec):
    pass
        
