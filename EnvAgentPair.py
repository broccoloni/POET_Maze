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
                 mutpower = 0.01,
                 lr = 0.01,
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
                 numactions = 10):
        

        
        #from inputs
        self.y               = y
        self.x               = x
        self.p_width         = p_width
        self.popsize         = popsize
        self.testsize        = testsize
        self.numagents       = numagents
        self.mutpower        = mutpower
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
        self.numactions      = numactions
        
        if framespercell is not None:
            self.maxframes = int(self.y * self.x * self.framespercell)
            
        #can set sample freq to eg. 0.1 to sample every 10% of maxframes
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
        self.curagentgen        = 0
        self.agentarch          = [0] #keeps track agent transfers and when
        self.agent              = None
        self.id                 = 0
        self.avgframestofindend = []
        self.avgframesused      = []
        self.avgcellsvisited    = []
        self.noveltyarchive     = np.array([])
        self.novelnets          = []
        self.curgen             = 0
        self.curera             = 0
        self.mutations          = None
        self.distances          = None
        self.parent             = 0
        self.children           = []
        self.solved             = False
        self.inANNECS           = False
        self.active             = True
        self.bestagentupdated   = False
        self.validparent        = False
        self.eapscores          = [[0,0,0]]
        self.bestagent          = [0,0,0] #era, gen, mut
        self.path               = "./"
        
        self.agent = self.updateagent()
        self.env.reset(numagents = self.numagents,
                       obssize = self.obssize,
                       exitfoundreward = self.exitfoundreward,
                       render = False)
        
    def trainstep(self, n = 1):
        agent = self.updateagent()
        muts,fstfend,fsused,csvisited,behaviours = self.ESstep(agent)
        self.avgframestofindend.append(np.mean(fstfend))
        self.avgframesused.append(np.mean(fsused))
        self.avgcellsvisited.append(np.mean(csvisited))
        bestind = self.findbestagent(fstfend,csvisited,fsused)
        distances = self.behaviourdist(behaviours)
        self.calculatenovelty(distances,behaviours,fstfend,csvisited,fsused)
        
        #update the best agent
        bestscore = self.score(fstfend[bestind],csvisited[bestind],fsused[bestind])
        if self.betterscore(bestscore,self.eapscores[self.id]):
            self.bestagent = [self.curera,self.curgen,bestind]
            self.eapscores[self.id] = bestscore
            self.bestagentupdated = True
            if bestscore[-1] > 0:
                self.solved = True
            
        if bestscore[0] >= 0.5:
            self.validparent = True
                              
        #update the stored mutations
        muts = muts.reshape(1,-1)
        if self.mutations is None:
            self.mutations = muts
        else:
            self.mutations = np.append(self.mutations,muts,axis = 0)
            
        if self.distances is None:
            self.distances = np.ones((1,self.popsize))
        else:
            self.distances = np.append(self.distances,
                                       distances.reshape(1,-1),
                                       axis = 0)
        #update curgen
        self.curgen += 1
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
                    self.novelnets.append([self.curgen,ind,fstfend[ind],csvisited[ind],fsused[ind]])
        else:
            #first iteration, just get some behaviours and travel in some direction
            unique_behaviours, inds = np.unique(behaviours, axis = 0, return_index = True)
            self.noveltyarchive = unique_behaviours
            for ind in inds:
                self.novelnets.append([self.curgen,ind,fstfend[ind],csvisited[ind],fsused[ind]])
            
        #reduce archive size if it gets too large
        if len(self.noveltyarchive) > self.maxarchsize:
            self.noveltyarchive = self.noveltyarchive[-self.maxarchsize:]
        return
        
    def ESstep(self,agent,seed = None,return_best_score = False):
        if seed is None:
            seed = self.seed+self.curgen#to have dif muts at each step
        np.random.seed(seed) 
        muts = np.random.randint(0,almightyint,size = self.popsize)
        fstfend = []
        fsused = []
        csvisited = []
        self.htmlvids = []
        behaviours = np.array([])
                
        renders = [True if i in self.agentstorender else False for i in range(self.popsize)]
        for i in range(self.popsize):
            mut = muts[i]
            mutagent = deepcopy(agent)
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
                                      render = render)
        for t in range(self.maxframes):
            actions = []
            obs = processobs(observations)
            for i in range(self.numagents):
                #Forward pass
                probs = agent.forward(obs[:,i,:,:])

                #greedy action
                action = torch.argmax(probs)
                actions.append(action)

            #Take actions
            observations, rewards, done = self.env.step(actions)

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
            print(self.y,self.x,dif,t,self.maxframes,self.samplefreq,extra.shape)
            behaviour = np.append(behaviour,np.ones(dif*self.numagents)*self.env.endind)
        cellsvisited = len(np.unique(behaviour))
        return framestofindend, framesused, cellsvisited, behaviour
    
    def findbestagent(self,fstfend,csvisited,fsused):
        bestmut = 0
        fsused = np.array(fsused)
        minfsusedinds = np.where(fsused == fsused.min())[0]
        if len(minfsusedinds) == 1:
            return minfsusedinds[0]
        fstfend = np.array(fstfend)
        minfstfendinds = np.where(fstfend[minfsusedinds] == fstfend[minfsusedinds].min())[0]
        return minfsusedinds[minfstfendinds[0]]
            
    def getagent(self, gen, mutind,era = None,path = None): 
        if era is None:
            #right side to include 0
            era = np.searchsorted(self.agentarch,gen,side = 'right')-1 
            
        if era == self.curera:
            mutations = self.mutations
            distances = self.distances

        else:
            if path is None:
                path = self.path
            print("loading previous era:",era,"(current era is {})".format(self.curera))
            mutations,distances = self.loadmutdist(path,era)
                
        torch.manual_seed(self.agentseed)
        agent = Agent(inshape = [self.obssize,self.obssize])
        
        for i in range(gen):
            muts  = mutations[i]
            coefs = distances[i] * self.lr / (self.popsize * self.mutpower) 
            agent.mutate(coefs,muts)
        
        if mutind is not None:
            mut = self.mutations[gen,mutind]
            agent.mutate(self.mutpower, mut)
        return agent  
   
    def updateagent(self):
        if self.curgen == 0 and self.curera == 0 and self.curagentgen == 0:
            torch.manual_seed(self.agentseed)
            self.agent = Agent(inshape = [self.obssize,self.obssize],outsize = self.numactions)
       
        if self.curgen == self.curagentgen:
            return self.agent

        for i in range(self.curagentgen,self.curgen):
            muts = self.mutations[i]
            coefs = self.distances[i] * self.lr / (self.popsize * self.mutpower)
            
            self.agent.mutate(coefs,muts)
        self.curagentgen = self.curgen
        return self.agent
      
    def getbestagent(self):
        #-1 to gen because we update to right before gen, then apply mutation
        return self.getagent(self.bestagent[1], self.bestagent[2], era = self.bestagent[0])
    
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
    
    def betterscore(self,newscore,score):
        if newscore[2] > score[2]:
            return True
        elif newscore[2] == score[2] and newscore[0] > score[0]:
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
    
    def renderagent(self,gen,mutind):
        agent = self.getagent(gen,mutind)
        ftfend, fused, cvisited,behaviour = self.findbehaviour(agent,render = True)
        print("Frames to find end:\t[{}/{}]".format(ftfend+1,self.maxframes))
        print("Frames used:\t\t[{}/{}]".format(fused+1,self.maxframes))
        print("Cells visited:\t[{}/{}]".format(cvisited,self.y*self.x))
        print("Score:\t\t",self.score(ftfend,cvisited,fused))
        print("Behaviour:\n",behaviour)
        print("Generating animation...",end = "\t")
        vid = self.env.makegif()
        print("done!")
        return vid
    
    def rendertopagent(self):
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
    
    def renderarchiveagent(self,agentind):
        agent = self.novelnets[agentind]
        gen,mutind = agent[0],agent[1]
        return self.renderagent(gen,mutind)
        
    def __copy__(self):
        eap = EnvAgentPair(self.y,self.x,
                           seed            = self.seed,
                           p_width         = self.p_width,
                           popsize         = self.popsize,
                           testsize        = self.testsize,
                           numagents       = self.numagents,
                           mutpower        = self.mutpower,
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
        
        eap.agentarch         = deepcopy(self.agentarch)       
        eap.agent              = deepcopy(self.agent)
        eap.id                 = self.id
        eap.avgframestofindend = deepcopy(self.avgframestofindend)
        eap.avgframesused      = deepcopy(self.avgframesused)
        eap.avgcellsvisited    = deepcopy(self.avgcellsvisited)
        eap.noveltyarchive     = deepcopy(self.noveltyarchive)
        eap.novelnets          = deepcopy(self.novelnets)
        eap.curgen             = self.curgen
        eap.curera             = self.curera
        eap.mutations          = deepcopy(self.mutations)
        eap.distances          = deepcopy(self.distances)
        eap.parent             = self.parent
        eap.children           = deepcopy(self.children)
        eap.solved             = self.solved
        eap.inANNECS           = self.inANNECS
        eap.active             = self.active
        eap.validparent        = self.validparent
        eap.bestagentupdated   = self.bestagentupdated
        eap.eapscores          = deepcopy(self.eapscores)
        eap.bestagent          = deepcopy(self.bestagent)
        eap.path               = self.path
        
        eap.agent = eap.updateagent()
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
                    mutpower = None,
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
                    numactions = None):

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
        if mutpower is not None:
            self.mutpower = mutpower
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
        if numactions is not None:
            self.numactions = numactions

        if framespercell is not None:
            self.maxframes = int(self.y * self.x * self.framespercell)

        #can set sample freq to eg. 0.1 to sample every 10% of maxframes
        if self.samplefreqpc is not None:
            self.samplefreq = max(int(self.maxframes * self.samplefreqpc),1)

        return

    def storemutdist(self,path):
        self.path = path
        file = open(path+"eap{}_mutdist_era{}.pickle".format(self.id,self.curera),'wb+')
        file.write(pickle.dumps([self.mutations,self.distances]))
        file.close()
        return
    
    def loadmutdist(self,path,era):
        file = open(path+"eap{}_mutdist_era{}.pickle".format(self.id,era),'rb')
        data = file.read()
        file.close()
        mutations,distances = pickle.loads(data)
        return mutations,distances
    
    def save(self,path):
        self.path = path
        file = open(path+'eap{}.pickle'.format(self.id),'wb+')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return 
    
    def load(self,path,eapid,verbose = False):
        if verbose:
            print("loading EnvAgentPair id {} from {}".format(eapid,path))
        file = open(path+'eap{}.pickle'.format(eapid),'rb')
        data = file.read()
        file.close()
        self.__dict__ = pickle.loads(data)
        self.updateagent()
        return




