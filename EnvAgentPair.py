#!/usr/bin/env python
# coding: utf-8
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors as NN
import torch
from Agent import *
from Maze import *
import pickle
almightyint = 6671111 #call pizza pizza hey hey hey!

class EnvAgentPair():
    def __init__(self,y,x,
                 seed = almightyint,
                 p_width = 5,
                 popsize = 20,
                 testsize = 1,
                 numagents = 10,
                 maxmutpower = 0.1,
                 minmutpower = 0.001,
                 mutdecay = 0.99,
                 lr = 0.001,
                 samplefreq = 50,
                 samplefreqpc = 0.1,
                 k = 10,
                 maxframes = 1000,
                 framespercell = None,
                 noveltythresh = 0.2,
                 maxarchsize = 1000,
                 obssize = 9, 
                 exitfoundreward = 1,
                 agentstorender = [],
                 interval_delay = 100, 
                 repeat_delay = 200, 
                 psizemut = 4,
                 nummuts = 3,
                 mazemutpower = 0.9,
                 numreps = 5):
        
        #from inputs
        self.y               = y
        self.x               = x
        self.p_width         = p_width
        self.popsize         = popsize
        self.testsize        = testsize
        self.numagents       = numagents
        self.maxmutpower     = maxmutpower
        self.minmutpower     = minmutpower
        self.mutdecay        = mutdecay
        self.mutpower        = maxmutpower
        self.lr              = lr
        self.samplefreq      = samplefreq
        self.samplefreqpc    = samplefreqpc
        self.k               = k
        self.maxframes       = maxframes
        self.framespercell   = framespercell
        self.seed            = seed
        self.noveltythresh   = noveltythresh
        self.maxarchsize     = maxarchsize
        self.obssize         = obssize
        self.exitfoundreward = exitfoundreward
        self.psizemut        = psizemut
        self.nummuts         = nummuts
        self.mazemutpower    = mazemutpower
        self.numreps         = numreps
        
        if framespercell is not None:
            self.maxframes = int(self.y * self.x * self.framespercell)
            
        #can set samplefreqpc (per cell) to eg. 0.1 to sample every 10% of maxframes
        if samplefreqpc is not None:
            self.samplefreq = max(int(self.maxframes * samplefreqpc),1)

        #rendering video
        self.agentstorender  = agentstorender
        self.htmlvids        = []
        self.interval_delay  = interval_delay
        self.repeat_delay    = repeat_delay
        
        #internal values
        self.env                = Maze(y,x,p_width = self.p_width, seed = seed)
        self.agentseed          = seed+1
        self.agentarch          = [0] #keeps track agent transfers and bestagent updates
        self.agent              = None
        self.bestagent          = None
        self.id                 = 0
        self.avgframestofindend = []
        self.avgframesused      = []
        self.avgcellsvisited    = []
        self.noveltyarchive     = np.array([])
        self.novelnets          = []
        self.curgen             = 0
        self.eragen             = 0
        self.curera             = 0
        self.parent             = 0
        self.children           = []
        self.solved             = False
        self.inANNECS           = False
        self.active             = True
        self.bestagentupdated   = False
        self.validparent        = False
        self.eapscores          = [[0,0,0]]
        self.path               = "./"
        
        torch.manual_seed(self.agentseed)
        self.agentdict = Agent(inshape = [self.obssize,self.obssize],numreps = self.numreps).state_dict()
        self.bestagentdict = deepcopy(self.agent)
        self.env.reset(numagents = self.numagents,
                       obssize = self.obssize,
                       exitfoundreward = self.exitfoundreward,
                       render = False)
        
    def trainstep(self, n = 1):
        if self.eragen == 0 or self.bestagentupdated:
            self.mutpower = self.maxmutpower
        
        muts,fstfend,fsused,csvisited,behaviours = self.ESstep(self.agentdict)
        self.avgframestofindend.append(np.mean(fstfend))
        self.avgframesused.append(np.mean(fsused))
        self.avgcellsvisited.append(np.mean(csvisited))
        distances = self.behaviourdist(behaviours)
        self.calculatenovelty(distances,behaviours,fstfend,csvisited,fsused)
        distances = distances.flatten()
        
        #update the best agent
        scores = np.array([self.score(fstfend[i],csvisited[i],fsused[i]) for i in range(self.popsize)])
        bestind = self.findbestscore(scores)
        bestscore = scores[bestind]
#         print("all scores")
#         for i,score in enumerate(scores):
#             print(i,score)
#         print("best ind:",bestind)      
        if self.beatsbestagent(bestscore):
            newbestagent = Agent(inshape = [self.obssize,self.obssize],numreps = self.numreps)
            newbestagent.load_state_dict(deepcopy(self.agentdict))
            newbestagent.mutate(self.mutpower,muts[bestind])

            self.bestagentdict = deepcopy(newbestagent.state_dict())
            del newbestagent
            
            self.eapscores[self.id] = bestscore
            self.bestagentupdated = True
            if bestscore[-1] > 0:
                self.solved = True
            
        if bestscore[0] >= 0.5:
            self.validparent = True
            
        #update mutation power
        sumftfend = np.sum(scores[:,0])
        if sumftfend == 0:
            self.mutpower = min(self.maxmutpower,self.mutpower/self.mutdecay)
        else:
            self.mutpower = max(self.minmutpower,self.mutpower*self.mutdecay)
            
        #store the muts and distances
        self.appendmutdist(muts,distances)
            
        #update the agent
        self.updateagent(muts,distances)
            
        #update curgen
        self.curgen += 1
        self.eragen += 1
        if n > 1:
            self.trainstep(n = n-1)
        return
        
    def behaviourdist(self,behaviours):
        if len(self.noveltyarchive) > 0: 
            #calculate k nearest neighbors distance
            k = min(self.k,len(behaviours)-1)
            #k+1 since it will always be closest to itself
            nn = NN(metric = 'hamming',n_neighbors = k+1).fit(np.append(self.noveltyarchive,
                                                                        behaviours,
                                                                        axis = 0)) #hamming:(sum of 0 if equal, else 1)/#neighbors
            distances,_ = nn.kneighbors(behaviours) 
            distances = distances.sum(axis = 1)

#             print("novelty archive size:",len(self.noveltyarchive))
#             print("behaviours:\n")
#             for i,behaviour in enumerate(behaviours[len(self.noveltyarchive):]):
#                 print("{:.3f}\t".format(distances[i]),behaviour)
        else:
            distances = np.ones((1,self.popsize))
        return distances
    
    def calculatenovelty(self,distances,behaviours,fstfend,csvisited,fsused):
        if len(self.noveltyarchive) > 0:
            #Add to novelty archive if novelty is found
            k = min(self.k,len(behaviours)-1)
            noveltyinds = [i for i,d in enumerate(distances) if d/k >= self.noveltythresh]      
            if len(noveltyinds) != 0:
                newbehaviours = behaviours[noveltyinds]
                self.noveltyarchive = np.append(self.noveltyarchive,
                                                newbehaviours,
                                                axis = 0)
                #store how we got the model, and its performance
                for ind in noveltyinds:
                    self.novelnets.append([self.eragen,ind,fstfend[ind],csvisited[ind],fsused[ind]])
        else:
            #first iteration, just get some behaviours and travel in some direction
            unique_behaviours, inds = np.unique(behaviours, axis = 0, return_index = True)
            self.noveltyarchive = unique_behaviours
            for ind in inds:
                self.novelnets.append([self.eragen,ind,fstfend[ind],csvisited[ind],fsused[ind]])
            
        #reduce archive size if it gets too large
        if len(self.noveltyarchive) > self.maxarchsize:
            self.noveltyarchive = self.noveltyarchive[-self.maxarchsize:]
        return
        
    def ESstep(self,agentdict,seed = None,return_best_score = False):
        if seed is None:
            seed = self.seed+almightyint * self.curera + self.curgen#to have dif muts at each step
        np.random.seed(seed) 
        muts = np.random.randint(0,almightyint,size = self.popsize)
        fstfend = []
        fsused = []
        csvisited = []
        self.htmlvids = []
        behaviours = np.array([])
                        
        mutagent = Agent(inshape = [self.obssize,self.obssize],numreps = self.numreps)
        renders = [True if i in self.agentstorender else False for i in range(self.popsize)]
        for i in range(self.popsize):
            mut = muts[i]
            mutagent.load_state_dict(deepcopy(self.agentdict))
            mutagent.mutate(self.mutpower,mut)
            ftfend, fused, cvisited, behaviour = self.findbehaviour(mutagent,render = renders[i])
            fstfend.append(ftfend)
            fsused.append(fused)   
            csvisited.append(cvisited)
            if renders[i]:
                self.htmlvids.append(self.env.makegif(interval_delay=self.interval_delay,repeat_delay=self.repeat_delay))
            behaviour = behaviour.reshape(1,-1)
            if len(behaviours) == 0:
                behaviours = behaviour
            else:
                behaviours = np.append(behaviours, behaviour, axis = 0)
        
        if return_best_score:
            bestind = self.findbestagent(fstfend,csvisited,fsused)
            bestftfend = fstfend[bestind]
            bestfused  = fsused[bestind]
            bestcvisited = csvisited[bestind]
            return self.score(bestftfend,csvisited,bestfused)
        return muts,fstfend,fsused,csvisited,behaviours
                                       
    def findbehaviour(self, agent, render = False):
        #Initialize behaviour
        #The behaviour will be the sequence of cells the agent is in,
        #sampled every sample_freq samples
        framestofindend = 0
        framesused = 0
        behaviour = []
        foundend = False
        observations = self.env.reset(numagents = self.numagents,
                                      obssize = self.obssize,
                                      exitfoundreward = self.exitfoundreward,
                                      render = render,
                                      numreps = self.numreps)

        agent.reset()
        for t in range(self.maxframes):
            movements = []
            reps = []
            obs = processobs(observations)
            for i in range(self.numagents):
                #Forward pass
                moveprobs,repprobs = agent.forward(obs[:,i,:,:])
                

                #greedy action
                movement = torch.argmax(moveprobs)
                movements.append(movement)
                
                rep = torch.argmax(repprobs)
                reps.append(rep)

            #Take actions
            observations, rewards, done = self.env.step(movements,reps)

            #store # of frames to find end
            if np.sum(rewards) != 0 and not done:
                framestofindend = t
                foundend = True
                
            if done:
                break
 
            #behaviour is sequence of cells that agents are in
            if t%self.samplefreq == 0:
                behaviour.extend(self.env.getagentcellinds())

        #keep track of # of frames used
        framesused = t
        if not foundend:
            framestofindend = t
                              
        #if finishing early, extend behaviour so all behaviours are the same length
        behaviour = np.array(behaviour).flatten()
        if t != self.maxframes-1:
            #how many frames early it finished
            dif = (self.maxframes - t)//self.samplefreq
            #make extra the ending cell
            extra = np.ones(dif*self.numagents)*self.env.endind
            behaviour = np.append(behaviour,np.ones(dif*self.numagents)*self.env.endind)
        cellsvisited = len(np.unique(behaviour))

        return framestofindend, framesused, cellsvisited, behaviour
    
    def findbestscore(self,scores):
        fusedscores = scores[:,2]
        ftfendscores = scores[:,0]
        
        maxinds = np.where(fusedscores == fusedscores.max())[0]
        if len(maxinds) > 1:
            return maxinds[np.argmax(ftfendscores[maxinds])]
        else:
            return maxinds[0]
        
    #THIS NEEDS TO BE UPDATED STILL
#     def getagent(self, gen, mutind,era = None,path = None): 
#         if era is None:
#             #right side to include 0
#             era = np.searchsorted(self.agentarch,gen,side = 'right')-1 
        
#         print("loading muts and dists for era:",era,"(current era is {})".format(self.curera))
#         mutations,distances = self.loadmutdist(path,self.id,era)
        
#         eap = EAP(2,2).load(self.eapid,path = self.path) #probably won't work like this, need to get start agent of era
#         agent = deepcopy(eap.agent)
#         del eap
        
#         for i in range(gen):
#             muts  = mutations[i]
#             coefs = distances[i] * self.lr / (self.popsize * self.mutpower) 
#             agent.mutate(coefs,muts)
        
#         if mutind is not None:
#             mut = self.mutations[gen,mutind]
#             agent.mutate(self.mutpower, mut)
#         return agent  
   
    def updateagent(self,muts,dists):
        for j in range(self.popsize):
            dist,mut = dists[j],muts[j]
            if dist != 0:
                for i,name in enumerate(self.agentdict):
                    torch.manual_seed(mut+i) #seed+i for each layer is still sampling from N,
                                              #it's just easier to do it for each layer individually
                    shape = self.agentdict[name].shape
                    self.agentdict[name] += dist * torch.empty(shape).normal_(mean=0,std=1)
        return
      
    def getbestagent(self):
        bestagent = Agent(inshape = [self.obssize,self.obssize],numreps = self.numreps)
        bestagent.load_state_dict(self.bestagentdict)
        return bestagent
    
    def mincriteriapass(self,score):
        ftfendscore = score[0]
        csvisitedscore = score[1]
        fusedscore = score[2]
        if ftfendscore >= 0.5 and fusedscore <= 0.5:
            return True
        return False
    
    def score(self,bestftfend,bestcellsvisited,bestfused):
        return [(self.maxframes-1-bestftfend)/(self.maxframes-1),
                bestcellsvisited/(self.y*self.x),
                (self.maxframes-1-bestfused)/(self.maxframes-1)]
    
    def beatsbestagent(self,score):
        bestscore = self.eapscores[self.id]
        if score[2] > bestscore[2]:
            return True
        elif score[2] == bestscore[2] and score[0] > bestscore[0]:
            return True
        else:
            return False
        
    def generatemutations(self,numtogenerate = 1):
        mutations = []
        for i in range(numtogenerate):
            generated = False
            attempts = 0
            seed = self.seed+attempts+self.curgen
            while not generated:
                eap = self.__copy__()
                sizemut = False
                
                #chance of mutating y
                np.random.seed(seed)
                p = np.random.random()
                if p < self.getpsizemut() or attempts >= 5:
                    eap.y += 1
                    sizemut = True
                seed += 1
                np.random.seed(seed)
                p = np.random.random()
                if p < self.getpsizemut() or attempts >= 5:
                    eap.x += 1
                    sizemut = True
                    
                seed += 1
                if sizemut:
                    eap.resetinputs() #to adjust framespercell/maxframes for new y and x
                    eap.env.mutate(eap.y,eap.x,0,0,seed)
                    newmaze = True #we will get a different maze than the parent
                else:
                    eap.env.mutate(eap.y,eap.x,eap.mazemutpower,eap.nummuts,seed)
                    newmaze = eap.env.isdifferent(self.env) #check if we get different maze than parent
                    
                #check if new maze is different from other mutations generated
                if newmaze:
                    for child in mutations:
                        if not eap.env.isdifferent(child.env):
                            newmaze = False
                            
                #if completely new maze append to mutations list
                if newmaze:
                    mutations.append(eap)
                    generated = True
                    
                attempts += 1
                seed += 1
                #also do decaying mutpower or that grows with no new behaviours found
        return mutations
    
    def getpsizemut(self):
        if self.psizemut >= 1:
            return min(1/(self.y*self.x)*self.psizemut,1)
        else:
            return self.psizemut
    
#     def renderagent(self,gen,mutind):
#         agent = self.getagent(gen,mutind)
#         ftfend, fused, cvisited,behaviour = self.findbehaviour(agent,render = True)
#         print("Frames to find end:\t[{}/{}]".format(ftfend+1,self.maxframes))
#         print("Frames used:\t\t[{}/{}]".format(fused+1,self.maxframes))
#         print("Cells visited:\t[{}/{}]".format(cvisited,self.y*self.x))
#         print("Score:\t\t",self.score(ftfend,cvisited,fused))
#         print("Behaviour:\n",behaviour)
#         print("Generating animation...",end = "\t")
#         vid = self.env.makegif()
#         print("done!")
#         return vid
    
    def renderbestagent(self):
        agent = self.getbestagent()
        ftfend,fused,cvisited,behaviour = self.findbehaviour(agent,render = True)
        print("Frames to find end:\t[{}/{}]".format(ftfend+1,self.maxframes))
        print("Frames used:\t\t[{}/{}]".format(fused+1,self.maxframes))
        print("Cells visited:\t[{}/{}]".format(cvisited,self.y*self.x))
        print("Score:\t\t",self.score(ftfend,cvisited,fused))
        print("Behaviour:\n",behaviour)
        print("Generating animation...",end = "\t")
        vid = self.env.makegif()
        print("done!")
        return vid
    
#     #THIS NEEDS TO BE UPDATED STILL
#     def renderarchiveagent(self,agentind):
#         agent = self.novelnets[agentind]
#         gen,mutind = agent[0],agent[1]
#         return self.renderagent(gen,mutind)
        
    def __copy__(self):
        eap = EnvAgentPair(self.y,self.x,
                           seed            = self.seed,
                           p_width         = self.p_width,
                           popsize         = self.popsize,
                           testsize        = self.testsize,
                           numagents       = self.numagents,
                           maxmutpower     = self.maxmutpower,
                           minmutpower     = self.minmutpower,
                           mutdecay        = self.mutdecay,
                           lr              = self.lr,
                           samplefreq      = self.samplefreq,
                           samplefreqpc    = self.samplefreqpc,
                           k               = self.k,
                           maxframes       = self.maxframes,
                           framespercell   = self.framespercell,
                           noveltythresh   = self.noveltythresh,
                           maxarchsize     = self.maxarchsize,
                           obssize         = self.obssize,
                           exitfoundreward = self.exitfoundreward,
                           agentstorender  = self.agentstorender,
                           interval_delay  = self.interval_delay,
                           repeat_delay    = self.repeat_delay,
                           psizemut        = self.psizemut,
                           nummuts         = self.nummuts,
                           mazemutpower    = self.mazemutpower)
        
        eap.agentarch          = deepcopy(self.agentarch)       
        eap.agentdict          = deepcopy(self.agentdict)
        eap.bestagentdict      = deepcopy(self.bestagentdict)
        eap.id                 = self.id
        eap.avgframestofindend = deepcopy(self.avgframestofindend)
        eap.avgframesused      = deepcopy(self.avgframesused)
        eap.avgcellsvisited    = deepcopy(self.avgcellsvisited)
        eap.noveltyarchive     = deepcopy(self.noveltyarchive)
        eap.novelnets          = deepcopy(self.novelnets)
        eap.curgen             = self.curgen
        eap.eragen             = self.eragen
        eap.curera             = self.curera
        eap.parent             = self.parent
        eap.children           = deepcopy(self.children)
        eap.solved             = self.solved
        eap.inANNECS           = self.inANNECS
        eap.active             = self.active
        eap.validparent        = self.validparent
        eap.bestagentupdated   = self.bestagentupdated
        eap.eapscores          = deepcopy(self.eapscores)
        eap.path               = self.path
        
        eap.env.reset(numagents = eap.numagents,
                      obssize = eap.obssize,
                      exitfoundreward = eap.exitfoundreward,
                      render = False)
        
        return eap

    def resetinputs(self,
                    y = None,
                    x = None,
                    seed = None,
                    p_width = None,
                    popsize = None,
                    testsize = None,
                    numagents = None,
                    maxmutpower = None,
                    minmutpower = None,
                    mutdecay = None,
                    lr = None,
                    samplefreq = None,
                    samplefreqpc = None,
                    k = None,
                    maxframes = None,
                    framespercell = None,
                    noveltythresh = None,
                    maxarchsize = None,
                    obssize = None,
                    exitfoundreward = None,
                    agentstorender = None,
                    interval_delay = None,
                    repeat_delay = None,
                    psizemut = None,
                    nummuts = None,
                    mazemutpower = None,
                    numreps = None):

        if y is not None:
            self.y = y
        if x is not None:
            self.x = x
        if p_width is not None:
            self.p_width = p_width
        if popsize is not None:
            self.popsize = popsize
        if testsize is not None:
            self.testsize = testsize
        if numagents is not None:
            self.numagents = numagents
        if maxmutpower is not None:
            self.maxmutpower = maxmutpower
        if minmutpower is not None:
            self.minmutpower = minmutpower
        if mutdecay is not None:
            self.mutdecay = mutdecay
        if lr is not None:
            self.lr = lr
        if samplefreq is not None:
            self.samplefreq = samplefreq
        if samplefreqpc is not None:
            self.samplefreqpc = samplefreqpc
        if k is not None:
            self.k = k
        if maxframes is not None:
            self.maxframes = maxframes
        if framespercell is not None:
            self.framespercell = framespercell
        if seed is not None:
            self.seed = seed
        if noveltythresh is not None:
            self.noveltythresh = noveltythresh
        if maxarchsize is not None:
            self.maxarchsize = maxarchsize
        if obssize is not None:
            self.obssize = obssize
        if exitfoundreward is not None:
            self.exitfoundreward = exitfoundreward
        if psizemut is not None:
            self.psizemut = psizemut
        if nummuts is not None:
            self.nummuts = nummuts
        if mazemutpower is not None:
            self.mazemutpower = mazemutpower
        if numreps is not None:
            self.numreps = numreps

        if framespercell is not None:
            self.maxframes = int(self.y * self.x * self.framespercell)

        #can set sample freq to eg. 0.1 to sample every 10% of maxframes
        if self.samplefreqpc is not None:
            self.samplefreq = max(int(self.maxframes * self.samplefreqpc),1)

        return

    def setupchild(self,ID,path):
        self.id                 = ID
        self.path               = path
        self.children           = []
        self.validparent        = False
        self.solved             = False
        self.inANNECS           = False
        self.bestagentupdated   = True
        self.noveltyarchive     = np.array([])
        self.avgframestofindend = []
        self.avgframesused      = []
        self.novelnets          = []
        self.agentarch          = []
        self.curgen             = 0
        self.eragen             = 0
        self.curera             = -1 #since we'll change it right away with tranfer
        return
    
    def appendmutdist(self,muts,dists):
        file = open(self.path+"eap{}_mutdist_era{}.pickle".format(self.id,self.curera),"ab+")
        file.write(pickle.dumps([muts,dists]))
        file.close()
        return
    
    def loadmutdist(self,path,eapid,era):
        file = open(path+"eap{}_mutdist_era{}.pickle".format(eapid,era),'rb')
        data = file.read()
        file.close()
        mutations,distances = pickle.loads(data) #this might cause an issue with new way muts and dists are being saved
        return mutations,distances
    
    def save(self,path = None):
        if path is None:
            path = self.path
        file = open(path+"eap{}.pickle".format(self.id),'wb+')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return 
    
    def load(self,eapid,path = None,verbose = False):
        if path is None:
            path = self.path
        if verbose:
            print("loading EnvAgentPair id {} from {}".format(eapid,path))
        file = open(path+'eap{}.pickle'.format(eapid),'rb')
        data = file.read()
        file.close()
        self.__dict__ = pickle.loads(data)
        return




