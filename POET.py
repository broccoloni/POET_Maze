import numpy as np
from EnvAgentPair import *
from sklearn.neighbors import NearestNeighbors as NN
import os
from tqdm import tqdm

class POET():
    def __init__(self, version, 
                 starty        = 2, 
                 startx        = 2, 
                 numreps       = 5, 
                 lr            = 0.001, 
                 maxsigma      = 0.1,
                 minsigma      = 0.001,
                 sigmadecay    = 0.99,
                 mutfreq       = 20, 
                 transferfreq  = 20, 
                 eps           = 0, 
                 framespercell = 40, 
                 samplefreq    = 50,
                 samplefreqpc  = 0.1,
                 seed          = 6671111, 
                 maxeaps       = 30, 
                 maxchildren   = 3, 
                 maxadmitted   = 6,
                 k             = 5, 
                 pataclips     = [[0.1,1],[0,0.8]],
                 minpatadist   = 0.1,
                 savefreq      = 50,
                 savedir       = "./",
                 verbose       = False):
        
        #from inputs
        self.starty        = starty
        self.startx        = startx
        self.numreps       = numreps
        self.lr            = lr
        self.maxsigma      = maxsigma
        self.minsigma      = minsigma
        self.sigmadecay    = sigmadecay
        self.mutfreq       = mutfreq
        self.transferfreq  = transferfreq
        self.eps           = eps #int from 0 - 100, then gets / 100 to be from 0 - 1. likelihood of choosing suboptimal transfer
        self.seed          = seed
        self.savefreq      = savefreq
        self.framespercell = framespercell
        self.samplefreq    = samplefreq
        self.samplefreqpc  = samplefreqpc
        self.maxeaps       = maxeaps
        self.maxchildren   = maxchildren
        self.maxadmitted   = maxadmitted
        self.k             = k
        self.pataclips     = pataclips
        self.minpatadist   = minpatadist
        self.savedir       = savedir
        self.verbose       = verbose
        
        #internal initializations
        self.curiters     = 0
        self.ANNECS       = []
        self.activeeaps   = []
        self.transferarch = []
        
        #for saveing and loading
        self.version = version #to have multiple runs for statistical significance
        if not os.path.isdir(self.savedir):
            os.mkdir(self.savedir)
        if not os.path.isdir(self.savedir+"POETsaves/"):
            os.mkdir(self.savedir+"POETsaves/")
        self.path = self.savedir + "POETsaves/poetv{}eps{}reps{}/".format(self.version,self.eps,self.numreps)
        self.eappath  = self.path+"eaps/"
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        if not os.path.isdir(self.eappath):
            os.mkdir(self.eappath)
    
    def run(self,n):
        for j,i in enumerate(range(self.curiters,self.curiters+n)):
            if self.verbose:
                self.paddedprint(f"Iteration: {i}")
        
            #create first agent on first iteration
            if i == 0:
                self.activeeaps.append(0)
                firsteap = EnvAgentPair(self.starty,
                                        self.startx,
                                        seed = self.seed,
                                        maxmutpower = self.maxsigma,
                                        minmutpower = self.minsigma,
                                        mutdecay = self.sigmadecay,
                                        lr = self.lr,
                                        framespercell = self.framespercell,
                                        samplefreqpc = self.samplefreqpc,
                                        numreps = self.numreps)
                firsteap.setupchild(0,self.eappath)
                firsteap.curera = 0
                firsteap.save()
                del firsteap
                
            #saving
            if j > 0 and i % self.savefreq == 0:
                if self.verbose:
                    self.paddedprint(f"Saving iteration: {i}")
                self.save()
                
            #mutate environments
            if i > 0 and i % self.mutfreq == 0:
                if self.verbose:
                    self.paddedprint("Mutating environments")
                self.mutate_envs()
                
            #training
            if self.verbose:
                self.paddedprint("Training")
            for i,eapid in enumerate(self.activeeaps):
                if self.verbose:
                    print(f"[{i+1}/{len(self.activeeaps)}]",end = "\r")
                self.traineap(eapid)
            if self.verbose:
                print()
            
            #agent transfers
            if len(self.activeeaps) > 1 and i % self.transferfreq == 0:
                if self.verbose:
                    self.paddedprint("Attempting transfers")
                transfers = []
                for eapid in self.activeeaps:
                    if self.verbose:
                        print("computing transfer to env",eapid)
                    eap = self.geteapfromid(eapid)
                    passers, scores = self.evaluate_candidates(eap)
                    if len(passers) > 0:
                        passer = self.choose_transfer_agent(scores)
                        newagentid = passers[passer]
                        transfers.append([eapid,newagentid])
                        self.transferarch.append([eapid,newagentid,self.curiters])
                    del eap
                self.transfer_agents(transfers)
            
            if self.verbose:
                self.paddedprint("Best agents")
                for eapid in self.activeeaps:
                    eap = self.geteapfromid(eapid)
                    print("eap",eap.id,"score",eap.eapscores[eap.id])
                    del eap
            self.curiters += 1
        return
    
    def mutate_envs(self):
        #find valid parents
        validparents = []
        for eapid in self.activeeaps:
            eap = self.geteapfromid(eapid)
            if eap.validparent:
                validparents.append(eapid)
            del eap
        if self.verbose:
            print("valid parents:",validparents)
        if len(validparents) == 0:
            return
        
        #spawn children of parents
        children = self.env_reproduce(validparents)
        
        #check min criteria
        passing_children, allagentids = self.childmincriteria(children)
        children = [children[i] for i in passing_children]

        if len(children) == 0:
            return

        #rank by novelty
        if self.verbose:
            print("calculating pataecs")
        #get current, archived, and child environmental characterizations
        pataecs, childpataecs, scoresonchildren = self.getallpataecs(children)
        
        #choose children based on sufficient novelty of agent behaviour
        admitted = 0
        passing_children = []
        childinds = [i for i in range(len(childpataecs))]
        while admitted < self.maxadmitted-1:
            #calculate novelty of behaviour in environments
            childnoveltyorder = self.findchildorder(pataecs,childpataecs)
            if len(childnoveltyorder) == 0: #occurs when agent behaviours aren't novel enough
                break
                
            #get index of child
            passing_child = childinds.pop(childnoveltyorder[0])
            passing_children.append(passing_child)
            admitted += 1

            #adjust pataecs so we can find the next most novel env given the new set of envs
            patainds = [i for i in range(len(childpataecs))]
            newpataind = patainds.pop(childnoveltyorder[0]) #makes it easier to get new child pataecs
            if len(patainds) > 0:
                pataecs = np.append(pataecs,childpataecs[newpataind].reshape(1,-1),axis = 0)
                childpataecs = childpataecs[patainds]
            else:
                break
        
        if self.verbose:
            print(f"{admitted} environment(s) added")
        
        if admitted > 0:
            for eapid in range(self.activeeaps[-1]+1):
                #extend eapscores in  
                eap = self.geteapfromid(eapid)
                
                #update scores of new children for all prev envs
                for childind in passing_children:
                    eap.eapscores.append(deepcopy(scoresonchildren[childind][eapid]))
                eap.save()
                del eap

        transfers = []
        for i,childind in enumerate(passing_children):
            child = children[childind]
            
            child.setupchild(self.activeeaps[-1]+1,self.eappath)

            #add children to active environments
            self.activeeaps.append(child.id)
            
            #append child id to parents children
            parent = self.geteapfromid(child.parent)
            parent.children.append(child.id)
            del parent

            #get passing agents, and their scores on new child
            passers = allagentids[childind]
            scores = [scoresonchildren[childind][passer] for passer in passers]
            
            #choose new agent for child environment
            passer = self.choose_transfer_agent(scores)
            newagentid = passers[passer]
            
            #set up transfer
            transfers.append([child.id,newagentid])
            self.transferarch.append([child.id,newagentid,self.curiters])
            
            #save child
            child.save()

        #delete children
        for child in children:
            del child

        #transfer the best agents into them
        self.transfer_agents(transfers)
            
        #save and remove eaps if at capacity
        numactiveeaps = len(self.activeeaps)
        if numactiveeaps > self.maxeaps:
            num_removals = self.maxeaps - numactiveeaps
            for i in range(num_removals):
                eapid = self.activeeaps[i]
                eap = self.geteapfromid(eapid)
                eap.active = False
                eap.bestagentupdated = False
                eap.save()
                del eap
            self.activeeaps  = self.activeeaps[num_removals:]
        return
        
    def env_reproduce(self,validparents):
        allchildren = []
        for parent in validparents:
            eap = self.geteapfromid(parent)
            children = eap.generatemutations(numtogenerate = self.maxchildren)
            for j,child in enumerate(children):
                if self.verbose:
                    print("making child",j,"of eap",parent)
                child.id = -1
                child.parent = parent
                allchildren.append(child)
            del eap
        return allchildren
    
    def evaluate_candidates(self,eap):
        passers = []
        scores  = []
        for eapid in self.activeeaps:                
            #direct transfer
            score = self.getscore(eapid,eap)
            passer = eap.mincriteriapass(score)
            
#             #fine tuning transfer
#             np.random.seed(eapid+self.curiters)
#             seed = np.random.randint(0,almightyint)
#             agenteap = self.geteapfromid(eapid)
#             score = eap.ESstep(agenteap.getbestagent(), seed = seed, return_best_score = True)
#             passer = eap.mincriteriapass(score)

            if passer:
                passers.append(eapid)
                scores.append(score)
        return passers, scores
        
    def choose_transfer_agent(self,scores):        
        if len(scores) == 1:
            return 0
        np.random.seed(self.curiters + len(scores))
        p = np.random.random()
        arrscores = np.array(scores)
        ftfendscores = arrscores[:,0]
        fusedscores = arrscores[:,2]
        if p < 1-self.eps/100:
            maxinds = np.where(fusedscores == fusedscores.max())[0]
            if len(maxinds) > 1:
                return maxinds[np.argmax(ftfendscores[maxinds])]
            else:
                return maxinds[0]
        else:
            numpassers = len(ftfendscores)
            np.random.seed(self.curiters+len(scores)+1)
            return np.random.randint(0,numpassers)
        
    def transfer_agents(self,transfers,scores = None):
        #storing era before transfers ensures we don't overwrite and agent
        #before transfering to another environment
        if len(transfers) > 0:
            transfers = np.array(transfers)
            envids = transfers[:,0]
            agentids = transfers[:,1]
            
            agentdicts = []
            eapscores  = []
            for agentid in agentids:
                agenteap = self.geteapfromid(agentid)
                agentdicts.append(deepcopy(agenteap.bestagentdict))
                eapscores.append(deepcopy(agenteap.eapscores))
                del agenteap

            for i in range(len(transfers)):
                envid = envids[i]
                agentid = agentids[i]
                if envid != agentid:
                    if self.verbose:
                        print(f"transfering agent {agentid} to environment {envid}")
                    #load eaps
                    enveap = self.geteapfromid(envid)
                    
                    #increase era for enveap and update agent archive
                    enveap.curera += 1
                    enveap.agentarch.append(deepcopy(enveap.curgen))
                    enveap.eragen = 0
                    
                    #update the agents
                    enveap.agentdict = deepcopy(agentdicts[i])
                    enveap.bestagentdict = deepcopy(agentdicts[i])
                                        
                    #we know how this agent performs on other envs too
                    enveap.eapscores = deepcopy(eapscores[i])
                    
                    #save new eap
                    enveap.save()
                    del enveap
            del agentdicts
            del eapscores
        return
    
    def getscore(self,agentid,eap):
        if agentid == eap.id:
            return eap.eapscores[eap.id]
        else:
            agenteap = self.geteapfromid(agentid)
            
        if (not agenteap.bestagentupdated) and eap.id >= 0:
            score = agenteap.eapscores[eap.id]   
        else:
            ftfend,fused,cvisited,_ = eap.findbehaviour(agenteap.getbestagent())
            score = eap.score(ftfend,cvisited,fused)
            if eap.id >= 0:
                agenteap.eapscores[eap.id] = score
                agenteap.save()
        del agenteap
        return score
    
    def getpataec(self,eap, return_scores = False):
        if self.activeeaps[-1] == 0:
            if return_scores:
                return np.array([[0]]), [self.getscore(0,eap)]
            return np.array([[0]])
        
        #evaluate
        clippedscores  = []
        origscores = []
        allsolved = True
                
        for i in range(self.activeeaps[-1]+1):
            score = self.getscore(i,eap)
                
            if return_scores:
                origscores.append(score)
            
            if score[2] == 0:
                allsolved = False

            #clip
            score[0] = max(score[0],self.pataclips[0][0])
            score[0] = min(score[0],self.pataclips[0][1])
            score[2] = max(score[2],self.pataclips[1][0])
            score[2] = min(score[2],self.pataclips[1][1])
           
            clippedscores.append(score)

        if allsolved and not eap.inANNECS:
            self.ANNECS.append(eap.id)
            eap.inANNECS = True
            
        #rank
        arrscores = np.array(clippedscores)
        inds = arrscores[:,0].argsort() #first sort by ftfend
        scores = arrscores[inds]
        sortedinds = inds[arrscores[:,2].argsort(kind = 'mergesort')] #then sort by fused with order preserving method

        #normalize
        pataec = (sortedinds)/(sortedinds.max())
        pataec -= 0.5
        pataec = pataec.reshape(1,-1)
        if return_scores:
            return pataec,origscores
        return pataec
        
    def getallpataecs(self,children):
        if self.verbose:
            print("getting pataec for eap", 0)
        pataecs = self.getpataec(self.geteapfromid(0))
        for i in range(1,self.activeeaps[-1]+1):
            if self.verbose:
                print("getting pataec for eap", i)
            eap = self.geteapfromid(i)
            pataecs = np.append(pataecs,self.getpataec(eap),axis = 0)
            del eap

        scoresonchildren = []
        childpataecs, scoresonchild = self.getpataec(children[0],
                                                     return_scores = True)
        scoresonchildren.append(scoresonchild)
        for i,child in enumerate(children):
            if self.verbose:
                print("getting pataec for child",i)
            if i == 0:
                continue
            childpataec, scoresonchild = self.getpataec(child, 
                                                        return_scores = True)
            scoresonchildren.append(scoresonchild)
            childpataecs = np.append(childpataecs,childpataec,axis = 0)
        
        if self.verbose:
            print("setting best agent updated to false")
        
        for eapid in self.activeeaps:
            eap = self.geteapfromid(eapid)
            eap.bestagentupdated = False
            eap.save()
            del eap
        return pataecs, childpataecs, scoresonchildren
    
    def findchildorder(self,pataecs,newpataecs):        
        if self.activeeaps[-1] == 0:
            return np.arange(len(newpataecs))
        
        k = min(self.k,len(pataecs))
        nn = NN(n_neighbors = k).fit(pataecs)
        distances,_ = nn.kneighbors(newpataecs)
        distances = distances.sum(axis = 1)
        
        inds = np.arange(len(distances))[distances >= self.minpatadist]
        if len(inds) > 0:
            distances = distances[inds]
            neworder = np.argsort(distances)[::-1] #descending order
            inds = inds[neworder]
        
        return inds
    
    def childmincriteria(self,children):
        if self.verbose:
            print("Testing children minimal criteria")
        passing_children = []
        passing_agent_ids = []
        
        for i,child in enumerate(children):
            if self.verbose:
                print("child",i,"novelty:",end = "\t")
            newenv = True
            for eapid in range(self.activeeaps[-1]+1):
                eap = self.geteapfromid(eapid)
                newenv = eap.env.isdifferent(child.env)
                del eap
                if not newenv:
                    break
            if newenv:
                if self.verbose:
                    print("pass,\t minimum criteria: ",end = "\t")
                passers, scores = self.evaluate_candidates(child) #should keep track of these to save a bit of compute in pataec
                if len(passers) > 0:
                    if self.verbose:
                        print("pass")
                    passing_children.append(i)
                    passing_agent_ids.append(passers)
                else:
                    if self.verbose:
                        print("fail")
            else:
                if self.verbose:
                    print("fail")
        return passing_children, passing_agent_ids
    
    def geteapfromid(self,eapid):
        eap = EnvAgentPair(self.starty,
                           self.startx,
                           seed = self.seed,
                           maxmutpower = self.maxsigma,
                           minmutpower = self.minsigma,
                           mutdecay = self.sigmadecay,
                           lr = self.lr,
                           framespercell = self.framespercell,
                           samplefreqpc = self.samplefreqpc,
                           numreps = self.numreps)
        #eap.load(self.eappath,eapid,verbose = self.verbose) #too much printing
        eap.load(eapid,path = self.eappath)
        return eap
    
    def traineap(self,eapid):
        eap = self.geteapfromid(eapid)
        eap.trainstep()
        eap.save()
        del eap
        return
    
    def loadlastsave(self):
        if os.path.isdir(self.path):
            contents = os.listdir(self.path)
        else:
            contents = []

        saveiters = []
        for name in contents:
            if name[:9] == "poet_iter":
                start = name.split(".")[0]
                iternum = int(start[9:])
                saveiters.append(iternum)
        maxiter = 0
        if len(saveiters) > 0:
            maxiter = max(saveiters)
        if maxiter > 0:
            self.load(maxiter)
        return
    
    def save(self,path = None):
        if path is None:
            path = self.path
        if self.verbose:
            print("saving iteration {} to {}".format(self.curiters,path))
        file = open(path+'poet_iter{}.pickle'.format(self.curiters),'wb+')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return 
    
    def load(self,curiters,path = None):
        if path is None:
            path = self.path
        if self.verbose:
            print("loading iteration {} from {}".format(curiters,path))
        file = open(path+'poet_iter{}.pickle'.format(curiters),'rb')
        data = file.read()
        file.close()
        self.__dict__ = pickle.loads(data)
        return
    
    def paddedprint(self,mystr):
        padwidth = 3
        print()
        print("#"*(padwidth+1)+"#"*len(mystr)+"#"*(padwidth+1))
        print("#"*padwidth,mystr,"#"*padwidth)
        print("#"*(padwidth+1)+"#"*len(mystr)+"#"*(padwidth+1))
        return
    
