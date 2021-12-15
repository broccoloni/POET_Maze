import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

class Maze():
    def __init__(self,y,x,p_width = 5,seed = 6671111):
        #cell number meanings:
        # 0   - passage
        # 1   - wall
        # 2   - end of maze
        # 3-7 - agent representations
        
        #maze related params
        self.y             = y
        self.x             = x
        self.height        = y*(p_width+1)+1
        self.width         = x*(p_width+1)+1
        self.p_width       = p_width
        self.pathval       = 0
        self.wallval       = 1
        self.endval        = 2
        self.seed          = seed
        self.neighbor_adds = [[0,-1],[0,1],[-1,0],[1,0]] #look left, right, up, down
        self.startind      = 0
        self.endind        = 0
        self.maze          = None
        np.random.seed(self.seed)        
        self.pdist         = np.random.random(self.y*self.x)
#         self.pdist         = np.ones(self.y*self.x)

        #agent related params - need to be reset with new agent
        self.stepnum              = 0
        self.numagents            = 0
        self.obsrad               = 0
        self.exitfoundreward      = 0
        self.agentlocs            = []
        self.agentrepresentations = []
        self.agentmask            = None
        self.endfound             = False
        self.render               = False 
        self.ims                  = []
        self.paddedmaze           = None #here since amount of padding depends on obsrad
        self.reps                 = 5
        #generate the maze
        self.generate_kruskals()
        
    def generate_kruskals(self):
        seed = self.seed
        np.random.seed(seed)
#         self.startind = np.random.randint(0,self.y*self.x)   

        #generate maze and make all cells passages
        self.maze = np.zeros((self.height,self.width)) + self.wallval
        for ind in range(self.y*self.x):
            self.change_cell(ind,self.pathval)
        
        buckets = [[i] for i in range(self.y*self.x)]
        bucketps = deepcopy(self.pdist)
        bucketneighbors = [self.find_neighbors(i) for i in range(self.y*self.x)]
        bucketindmap = np.arange(self.y*self.x)
                
        for i in range(self.y*self.x-1):            
            #find minimum probability bucket indices
            bucketinds = np.where(bucketps == bucketps.min())[0]
                        
            #choose one of the minimum probability buckets
            np.random.seed(seed)
            bucket1 = np.random.choice(bucketinds,1)[0]
                        
            #find minimum probability bucket indices of bucket1 neighbors
            minneighborinds = np.where(bucketps[bucketindmap[bucketneighbors[bucket1]]] == bucketps[bucketindmap[bucketneighbors[bucket1]]].min())[0]
            minbucket1neighbors = bucketindmap[[bucketneighbors[bucket1][i] for i in minneighborinds]]
                        
            #choose one of the minimum probability neighbors
            np.random.seed(seed)
            bucket2 = np.random.choice(minbucket1neighbors,1)[0]
            
            if len(buckets[bucket1]) == 0:
                self.endind = buckets[bucket1][0]
                
            if len(buckets[bucket2]) == 0:
                self.endind = buckets[bucket2][0]
                        
            #find all neighbors of bucket1 in bucket2
            common_neighbors = np.intersect1d(bucketneighbors[bucket1],
                                              buckets[bucket2],
                                              assume_unique = True)
                                                                      
            #randomly choose neighbor
            np.random.seed(seed)
            cell1 = np.random.choice(common_neighbors,1)[0]
                        
            #find all neighbors of cell1 in bucket1
            cell1neighbors = self.find_neighbors(cell1)            
            candidates = [ind for ind in cell1neighbors if bucketindmap[ind] == bucket1]
                        
            #randomly choose one of the neighbors
            np.random.seed(seed)
            cell2 = np.random.choice(candidates,1)[0]
                        
            #connect the cells
            self.connect_cells(cell1,cell2)
            
            #join bucket1 and bucket2
            minbucket = min(bucket1,bucket2) #lower indexed bucket
            maxbucket = max(bucket1,bucket2)
            buckets[minbucket].extend(buckets.pop(maxbucket))
            
            #remove bucket1 from bucketps
            bucketps[minbucket] = bucketps[bucket2] #set minbucket prob to higher probability
            bucketps = np.delete(bucketps,maxbucket)
            
            #update bucketindmap
            bucketindmap[buckets[minbucket]] = minbucket
            bucketindmap[bucketindmap > maxbucket] -= 1

            #update bucketneighbors
            bucketneighbors[minbucket].extend(bucketneighbors.pop(maxbucket))
            bucketneighbors[minbucket] = list(np.unique(bucketneighbors[minbucket]))
            bucketneighbors[minbucket] = [ind for ind in bucketneighbors[minbucket] if ind not in buckets[minbucket]]
            
            seed += 1

            if i == 0:
                self.startind = bucket1
                
        #choose end of maze so that it's at least half the distance away from the startin both directions
        endindfar = False
        mindisty = self.y//2
        mindistx = self.x//2
        startcoords = self.itoc(self.startind)
        while not endindfar:
            np.random.seed(seed)
            endind = np.random.randint(0,self.y*self.x)
            endcoords = self.itoc(endind)
            disty = abs(startcoords[0] - endcoords[0])
            distx = abs(startcoords[1] - endcoords[1])
            if disty >= mindisty and distx >= mindistx:
                endindfar = True
                self.endind = endind
            seed += 1
        self.change_cell(self.endind,self.endval)
        
    def generate_prims(self):
        prob_weight = 1
        
        seed = self.seed
        np.random.seed(seed)
        self.startind = np.random.randint(0,self.y*self.x)       

        self.maze = np.zeros((self.height,self.width)) + self.wallval
        self.change_cell(self.startind,self.pathval)

        wall_list = self.cellfilter(self.find_neighbors(self.startind),self.wallval)
        numwalls = len(wall_list)
        recent_add = len(wall_list)

        while numwalls != 0:
            #randomly select a wall
            np.random.seed(seed)

            ps = np.exp([-prob_weight*self.dist(wall,self.startind) for wall in wall_list])
#             ps = np.ones(numwalls)
#             ps[-recent_add:] += self.prob_weight
            ps /= np.sum(ps)
            #w = np.random.randint(0,numwalls)
            w = np.random.choice(numwalls,1,p = ps)[0]
            wall = wall_list.pop(w)

            #calculate passages surrounding the wall
            passages = self.cellfilter(self.find_neighbors(wall),self.pathval)   

            #randomly select a passage
            np.random.seed(seed)
            selection = np.random.randint(0,len(passages))
            passage = passages[selection]

            #change cell to passage
            self.change_cell(wall,self.pathval)

            #change connection from passage to wall now turned passage to passage
            self.connect_cells(wall,passage)
            
            #append walls of new passage to wall list
            new_walls = self.cellfilter(self.find_neighbors(wall),self.wallval)
            recent_add = 0
            for new_wall in new_walls:
                if new_wall not in wall_list:
                    wall_list.append(new_wall)
                    recent_add += 1

            numwalls = len(wall_list)
            seed += 1

        #set end to last passage created
        self.endind = wall
        self.change_cell(self.endind,self.endval)

    def find_neighbors(self, ind): 
        neighbors = []
        coords = self.itoc(ind)
        for add in self.neighbor_adds:
            newy = coords[0]+add[0]
            newx = coords[1]+add[1]
            if self.isinside(newy,x = newx):
                neighbors.append(self.ctoi(newy,newx))
        return neighbors

    def cellfilter(self,inds,cellval):
        #assumes all inds in inds are in the maze
        filtered_cells = []
        for ind in inds:
            if self.cellsum(ind) == cellval*self.p_width*self.p_width:
                filtered_cells.append(ind)
        return filtered_cells
    
    def isinside(self,y,x = None, dims = None):
        if dims == None:
            dims = [self.y,self.x]
        
        #if x is None assume y is index
        if x is None:
            if y < 0 or y >= dims[0] * dims[1]:
                return False
            
        #if x is given assume coordinates of a cell
        else:
            if y < 0 or x < 0 or y >= dims[0] or x >= dims[1]:
                return False
        return True
    
    def ctoi(self,y,x, cols = None):
        if cols is None:
            cols = self.x
        return cols * y + x
    
    def itoc(self,ind, cols = None):
        if cols is None:
            cols = self.x
        return [ind//cols,ind%cols]
    
    def gridtocell(self, y, x = None):
        #if x is None assume index of grid position is given
        if x is None:
            coords = self.itoc(y,cols = self.x*(self.p_width+1)+1)
            y = coords[0]
            x = coords[1]
            cellcoords = self.gridtocell(y,x = x)
            return self.ctoi(cellcoords[0],cellcoords[1])
        else:
            return [(y-1)//(self.p_width+1),(x-1)//(self.p_width+1)]
        
    def celltogrid(self, y, x = None):
        #returns top left corner grid ind/coord for the ind/coord
        #of the cell given
        
        #if x is None assume index of cell position is given
        if x is None:
            coords = self.itoc(y)
            y = coords[0]
            x = coords[1]
            gridcoords = self.celltogrid(y,x = x)
            return self.ctoi(gridcoords[0],gridcoords[1],cols = self.x*(self.p_width+1)+1)
        else:
            gridy = y*(self.p_width+1)+1
            gridx = x*(self.p_width+1)+1
            return [gridy,gridx]
        
    def dist(self,ind1,ind2):
        #manhatten (L1) distance between two indicies
        coords1 = self.itoc(ind1)
        coords2 = self.itoc(ind2)
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])
    
    def connect_cells(self,ind1,ind2):
        #horizontal connection
        if abs(ind1-ind2) == 1 and self.x != 1:
            left = min(ind1,ind2)
            coords = self.itoc(left)
            ind = coords[0]*(self.x-1)+coords[1]
            self.change_horizconnection(ind,self.pathval)

        #vertical connection
        else:
            top = min(ind1,ind2) #for vertical connection, ind of top is same as ind of the connection
            self.change_vertconnection(top,self.pathval)
        return
        
    def change_cell(self,ind,val,curmaze = None,padding = 0):
        if curmaze is None:
            curmaze = self.maze
        coords = self.itoc(ind)
        y = coords[0]
        x = coords[1]
        curmaze[y*(self.p_width+1)+1+padding:(y+1)*(self.p_width+1)+padding,
                  x*(self.p_width+1)+1+padding:(x+1)*(self.p_width+1)+padding] = val
        return
        
    def cellsum(self,ind):
        coords = self.itoc(ind)
        y = coords[0]
        x = coords[1]
        cellsum = np.sum(self.maze[y*(self.p_width+1)+1:(y+1)*(self.p_width+1),
                                   x*(self.p_width+1)+1:(x+1)*(self.p_width+1)])
        return cellsum
    
    def change_horizconnection(self,ind,val):
        coords = self.itoc(ind,cols = self.x-1)
        y = coords[0]
        x = coords[1]
        self.maze[y*(self.p_width+1)+1:(y+1)*(self.p_width+1),
                  (x+1)*(self.p_width+1)] = val
        
    def change_vertconnection(self,ind,val):
        coords = self.itoc(ind)
        y = coords[0]
        x = coords[1]
        self.maze[(y+1)*(self.p_width+1),
                  x*(self.p_width+1)+1:(x+1)*(self.p_width+1)] = val
        
    def reset(self,numagents = 10,obssize = 7,exitfoundreward = 0, render = False,numreps = 5):
        #reset agent specific params
        self.stepnum = 0
        self.numagents = numagents
        self.obsrad = (obssize-1)//2
        self.exitfoundreward = exitfoundreward
        self.agentlocs = []
        self.agentrepresentations = []
        self.endfound = False
        self.render = render #if true, saves image of every step in self.ims and creates a gif when done
        self.ims = []
        self.numreps = numreps
        self.paddedmaze = np.pad(deepcopy(self.maze),
                                 ((self.obsrad,self.obsrad),(self.obsrad,self.obsrad)),'edge') #to make getting obs easier
        
        startcoords = self.itoc(self.startind)
        gridstart = self.celltogrid(startcoords[0],startcoords[1])
        self.agentlocs = []
        self.agentrepresentations = []
        
        np.random.seed(self.seed)
        positions = np.random.choice(self.p_width * self.p_width,
                                     self.numagents,
                                     replace = False)
                
        for i in range(self.numagents):
            agentiy = gridstart[0]+positions[i]//self.p_width
            agentix = gridstart[1]+positions[i]%self.p_width
            self.agentlocs.append([agentiy,agentix])
            self.agentrepresentations.append(3)
        
        self.updateagentmask()
        obs = self.getobs()
        self.renderer()
        return obs
        
    def step(self,movements,reps):
        #action meanings:
        #0  - move up
        #1  - move right
        #2  - move down
        #3  - move left
        #4+ - make representation 3+
        obsercations = []
        for i in range(len(movements)):
            movement = movements[i]
            representation = reps[i]
            
            agentpos = self.agentlocs[i]
            y,x = agentpos[0],agentpos[1]
            
            #movements
            #movement 0
            if movement == 0 and not self.isoccupied(y-1,x):
                self.agentlocs[i][0] -=  1
            #movement 1
            if movement == 1 and not self.isoccupied(y,x+1):
                self.agentlocs[i][1] += 1
            #movement 2
            if movement == 2 and not self.isoccupied(y+1,x):
                self.agentlocs[i][0] += 1
            #movement 3
            if movement == 3 and not self.isoccupied(y,x-1):
                self.agentlocs[i][1] -= 1
            #movement 4
            #do nothing

            #representations
            self.agentrepresentations[i] = representation+3 #+3 because other values are reserved for wall/passage/end
        
        self.updateagentmask()
        obs = self.getobs()
        rewards = self.getrewards()
        done = self.isdone()
        self.stepnum += 1
        self.renderer()
        return [obs,rewards,done]
    
    def isoccupied(self,y,x, curmaze = None):
        if curmaze is None:
            curmaze = self.maze
        if [y,x] in self.agentlocs:
            return True
            
        tileval = max(curmaze[y,x],self.agentmask[y+self.obsrad,x+self.obsrad])
        if tileval == self.pathval or tileval == self.endval:
            return False
        
        return True
        
    def updateagentmask(self):
        newmask = np.zeros((self.height,self.width)) 
        for agent in range(self.numagents):
            y,x = self.agentlocs[agent][0],self.agentlocs[agent][1]
            newmask[y,x] = self.agentrepresentations[agent]
        
        newmask = np.pad(newmask,((self.obsrad,self.obsrad),(self.obsrad,self.obsrad)),'edge')
        self.agentmask = newmask
        return 
    
    def getobs(self):
        observations = []
        for agent in range(self.numagents):            
            y = self.obsrad + self.agentlocs[agent][0]
            x = self.obsrad + self.agentlocs[agent][1]
            
            #bounds in padded maze
            u = y - self.obsrad #up
            d = y + self.obsrad #down
            l = x - self.obsrad #left
            r = x + self.obsrad #right
                        
            obs = deepcopy(self.paddedmaze[u:d+1,l:r+1])
            
            mask = deepcopy(self.agentmask[u:d+1,l:r+1])
            obs[self.agentmask[u:d+1,l:r+1] != 0] = self.agentmask[u:d+1,l:r+1][self.agentmask[u:d+1,l:r+1] != 0]
            
#             print("original obs:\n",obs)
            
            y = self.obsrad
            x = self.obsrad
            
            #set walls going outward from agent
            for i in range(self.obsrad-1): #radius out from agent
                #agent upward
                if obs[y-1-i,x] == self.wallval:
                    obs[y-2-i,x] = self.wallval
                
                #agent downward
                if obs[y+1+i,x] == self.wallval:
                    obs[y+2+i,x] == self.wallval
                    
                #agent leftward
                if obs[y,x-1-i] == self.wallval:
                    obs[y,x-2-i] = self.wallval
                    
                #agent rightward
                if obs[y,x+1+i] == self.wallval:
                    obs[y,x+2+i] = self.wallval                
                
            #set other walls not directly in line with agent
            for offset in range(1,self.obsrad+1): #offset from agent
                for i in range(1,self.obsrad): #radius out from agent
                    #agent upward offset left
                    if obs[y-i,x-offset] == self.wallval and obs[y-1-i,x-offset+1] == self.wallval:
                        obs[y-1-i,x-offset] = self.wallval

                    #agent upward offset right
                    if obs[y-i,x+offset] == self.wallval and obs[y-1-i,x+offset-1] == self.wallval:
                        obs[y-1-i,x+offset] = self.wallval
                        
                    #agent downward left
                    if obs[y+i,x-offset] == self.wallval and obs[y+1+i,x-offset+1] == self.wallval:
                        obs[y+1+i,x-offset] == self.wallval
                        
                    #agent downward right
                    if obs[y+i,x+offset] == self.wallval and obs[y+1+i,x+offset-1] == self.wallval:
                        obs[y+1+i,x+offset] == self.wallval

                    #agent leftward offset up
                    if obs[y-offset,x-i] == self.wallval and obs[y-offset+1,x-1-i] == self.wallval:
                        obs[y-offset,x-1-i] = self.wallval

                    #agent leftward offset down
                    if obs[y+offset,x-i] == self.wallval and obs[y+offset-1,x-1-i] == self.wallval:
                        obs[y+offset,x-1-i] = self.wallval
                        
                    #agent rightward offset up
                    if obs[y-offset,x+i] == self.wallval and obs[y-offset+1,x+1+i] == self.wallval:
                        obs[y-offset,x+1+i] = self.wallval    
                    
                    #agent rightward offset down
                    if obs[y+offset,x+i] == self.wallval and obs[y+offset-1,x+1+i] == self.wallval:
                        obs[y+offset,x+1+i] = self.wallval
                                            
            observations.append(obs)
#             print("final obs:\n",obs)
        return observations
        
    def getrewards(self):
        rewards = np.zeros(self.numagents)
        allonexit = True
        for i in range(self.numagents):
            gridind = self.ctoi(self.agentlocs[i][0],self.agentlocs[i][1],cols = self.x*(self.p_width+1)+1)
            if self.gridtocell(gridind) != self.endind:
                allonexit = False
            else:
                if not self.endfound:
                    self.endfound = True
                    rewards[i] += self.exitfoundreward
                    
        if allonexit:
            rewards += 100
        return rewards
    
    def isdone(self):
        for i in range(self.numagents):
            gridind = self.ctoi(self.agentlocs[i][0],self.agentlocs[i][1],cols = self.x*(self.p_width+1)+1)
            if self.gridtocell(gridind) != self.endind:
                return False
        return True
        
    def show(self,savepath = None): 
        if self.paddedmaze is not None:
            r = deepcopy(self.paddedmaze)
        else:
            r = deepcopy(self.maze)
        r[r == self.pathval] = 255
        r[r == self.wallval] = 0            
        r = r.reshape(r.shape[0],r.shape[1],1)
        g,b = deepcopy(r),deepcopy(r)
        
        #set start to green
        self.change_cell(self.startind,255,curmaze = g,padding = self.obsrad)
        self.change_cell(self.startind,0,curmaze = r,padding = self.obsrad)
        self.change_cell(self.startind,0,curmaze = b,padding = self.obsrad)
        #set end to red
        r[r == self.endval] = 255
        
        #add agents in blue
        if self.agentmask is not None: 
            #agents can be from 3 to 7
            #subtract so that agents are from 1 - 4, 
            #compute 255-obsshift - #agents * (agent representation)
            #minus a shift so that we can make a visible field of vision for each agent
            obsshift = 10
            
            #add field of vision
            for loc in self.agentlocs:
                #with the padding the bounds are:
                up = loc[0]
                do = loc[0]+2*self.obsrad+1
                le = loc[1]
                ri = loc[1]+2*self.obsrad+1
                
#                 r[up:do,le:ri] -= obsshift
#                 g[up:do,le:ri] -= obsshift
                b[up:do,le:ri] += obsshift
            
            #update agent location colour
            r[self.agentmask != 0] = 0
            g[self.agentmask != 0] = 0
            b[self.agentmask != 0] = ((255 - obsshift)-4*(self.agentmask[self.agentmask != 0]-2)).reshape(-1,1)
            
            

        im = np.stack((r,g,b),axis = 2).squeeze().astype(np.uint8)
        #make sure everything is in the range
        im[im < 0] = 0
        im[im > 255] = 255
        plt.imshow(im)
        if savepath is not None:
            plt.imsave(savepath,im)
        
        plt.show()
        return
        
    def getagentcellinds(self):
        agentinds = []
        for loc in self.agentlocs:
            gridy,gridx = loc[0],loc[1]
            celly,cellx = self.gridtocell(gridy,gridx)
            agentinds.append(self.ctoi(celly,x = cellx))
        return agentinds
        
    def renderer(self):
        if self.render:
            r = deepcopy(self.paddedmaze)
            r[r == self.pathval] = 255
            r[r == self.wallval] = 0
            r = r.reshape(r.shape[0],r.shape[1],1)
            g,b = deepcopy(r),deepcopy(r)

            #set start to green
            self.change_cell(self.startind,255,curmaze = g,padding = self.obsrad)
            self.change_cell(self.startind,0,curmaze = r,padding = self.obsrad)
            self.change_cell(self.startind,0,curmaze = b,padding = self.obsrad)
            #set end to red
            r[r == self.endval] = 255

            #add agents in blue
            if self.agentmask is not None: 
                #agents can be from 3 to 8
                #subtract so that agents are from 1 - 5, 
                #compute (255-obsshift) *(1 - agent rep / #representations)
                #minus a shift so that we can make a visible field of vision for each agent
                obsshift = 10

                #add field of vision
                for loc in self.agentlocs:
                    #with the padding the bounds are:
                    up = loc[0]
                    do = loc[0]+2*self.obsrad+1
                    le = loc[1]
                    ri = loc[1]+2*self.obsrad+1

    #                 r[up:do,le:ri] -= obsshift
    #                 g[up:do,le:ri] -= obsshift
                    b[up:do,le:ri] += obsshift

                #update agent location colour
                r[self.agentmask != 0] = 0
                g[self.agentmask != 0] = 0
                b[self.agentmask != 0] = ((255 - obsshift)*(1 - (self.agentmask[self.agentmask != 0]-2)/self.numreps)).reshape(-1,1)

            im = np.stack((r,g,b),axis = 2).squeeze().astype(np.uint8)
            #make sure everything is in the range
            im[im < 0] = 0
            im[im > 255] = 255
            self.ims.append(im)
        return
    
    def makegif(self, interval_delay = 100,repeat_delay = 200):
        fig,ax = plt.subplots()
        ims = [[ax.imshow(im)] for im in self.ims]
        ani = animation.ArtistAnimation(fig, ims, interval=interval_delay, blit=True,
                                repeat_delay=repeat_delay)
        return ani.to_html5_video()
    
    def mutate(self, newy, newx, mutpower, nummuts, mutseed):
        nummuts = int(nummuts) #for some reason it's changing this to float
        if newy != self.y or newx != self.x:
            self.pdist = self.pdist.reshape(self.y,self.x)
            ydif = newy - self.y
            xdif = newx - self.x
            lpad = xdif//2
            rpad = xdif - lpad
            tpad = ydif//2
            bpad = ydif - tpad
            self.pdist = np.pad(self.pdist,((tpad,bpad),(lpad,rpad)),'constant',constant_values = ((0,0),(0,0)))
            np.random.seed(mutseed)
            self.pdist[self.pdist == 0] = np.random.random(self.pdist[self.pdist == 0].shape)
            self.pdist = self.pdist.flatten()

            self.y      = newy
            self.x      = newx
            self.height = newy*(self.p_width+1)+1
            self.width  = newx*(self.p_width+1)+1
        
        np.random.seed(mutseed+1)
        mutlocs = np.random.randint(0,self.y*self.x,size = nummuts)
        
        
        np.random.seed(mutseed+2)
        self.pdist[mutlocs] += mutpower * np.random.random(nummuts)
        
        oldmaze = deepcopy(self.maze)
        self.generate_kruskals()
        self.reset(numagents       = self.numagents,
                   obssize         = 2*self.obsrad+1,
                   exitfoundreward = self.exitfoundreward, 
                   render          = self.render)
        return 
    
    def isdifferent(self,othermaze):
        if othermaze.y == self.y and othermaze.x == self.x:
            dif = np.sum(np.abs(self.maze - othermaze.maze))

            if dif == 0:
                return False
            return True
        return True
        
    def __str__(self):
        return f"Maze: {self.y} x {self.x}\nStarting location: {self.itoc(self.startind)}\nEnding location: {self.itoc(self.endind)}\n"

    def __repr__(self):
        return f"Maze: {self.y} x {self.x}\nStarting location: {self.itoc(self.startind)}\nEnding location: {self.itoc(self.endind)}\n"














