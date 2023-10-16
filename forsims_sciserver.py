"""
Created on Wed Jul 21 17:35:34 2021
Revised Late Fall + Winter of 2021-2022 to work on SciServer

@author: rlmcclure/jhunt/cfilion

"""


#%%
#python 3.8
import datetime
from astropy.units import equivalencies
import numpy as np
import galpy
try:
    from galpy.util import coords
except:
    from galpy.util import bovy_coords as coords
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import os
import glob
from scipy.signal import find_peaks

plt.rcParams.update({'font.size': 22})

class SimHandler(object):
    '''
    Dev by R L McClure to interact with an perform analysis on J Hunt's 2021 simulations

    This Class is designed to interact with the Bonsai simulation outputs. 


    Particles can be traced through the simulation and individual snaps can be loaded. 
    
    
    Some alterations by C. Filion to make it such that we can work with the data volume system
    
    of sciserver.

    There are also additional changes adapted from G.Lemson to speed up performance.
    
    note - C. Filion also added re-centering function, adapted from J Hunt. This 
    
    is an optional flag when loading a whole snap, and is an additional function that one can
    
    apply to a set of particle locations in time (an orbit). Note that to move into the bar frame, 
    
    the galaxy must be centered first.
    
    
    

    Contact rlmcclure@wisc.edu and/or cfilion@jhu.edu for updates or problems or friendship.
    
    ###
    For users of the original SimHandler class on the Flatiron cluster: C. Filion has 
    
    removed certain functionalities from the SimHandler that are more specialised 
    
    and related to R. McClure and C. Filion's projects, such that the SimHandler is
    
    now a generic class for using the simulations. 
    
    When initializing the class, you pass in the simulation name - this will ensure you are loading the simulation
    
    with the correct values for particle numbers etc
    
    This class no longer automatically computes 
    
    cylindrical coordinates, which speeds up performance! As such, commondtype no longer includes cylindrical coordinates
    
    
    ###
    
    '''


    def __init__(self, sim, *args,**kwargs):
        print('start init at: '+str(datetime.datetime.now()),flush=True)
        
        self.verbose = 0 #default verbosity is zero but can go up to like 4 or 5

        #path info defaults
        self.path = '/home/idies/workspace/SMUDGE/'
        self.db = 'data*'
        self.sim = str(sim)
        
        
        self.recenter = False
        
        if self.sim == 'M1':
            self.pointfpath = '/home/idies/workspace/SMUDGE/data11_03/M1pointers/' #path for pointer .npy files
        
            #initial values as determined by BN-2 simulation
            # there are 219,607,640 star particles in the disk, 21,960,680 in the bulge, 
            #and 153,599 in the dwarf satellite -- aka a total of 241,721,919 stars
            # there are 878431120 dark particles in big galaxy, 2880684 in the satellite
            self.ntot = 1123033723 
            self.ndark=881311804
            self.nstar= 241721919
            self.masscutoff = 400 #mass flag boundary between bulge and disk 
            
        if self.sim == 'M2':
            self.pointfpath = '/home/idies/workspace/SMUDGE/data11_03/M2pointers/' #path for pointer .npy files - none yet
            self.ntot = 1283033643
            self.ndark=1006801964
            self.nstar= 276231679
            self.masscutoff = 400 #mass flag boundary between bulge and disk 
        
        if self.sim == 'D2':
            self.pointfpath = '/home/idies/workspace/SMUDGE/data11_03/D2pointers/' #path for pointer .npy files

            self.ntot = 1279999360
            self.ndark=1003921280
            self.nstar= 276078080 
            self.masscutoff = 400 #mass flag boundary between bulge and disk 
            
        if self.sim == 'D1':
            self.pointfpath = '' #path for pointer .npy files
            print('note - there are no pointers or pattern speed files for this simulation')
            self.ntot = 1119999440
            self.ndark=878431120
            self.nstar= 241568320 
            self.masscutoff = 400 #mass flag boundary between bulge and disk 
            
        if self.sim != 'M1' and self.sim != 'M2' and self.sim != 'D2' and self.sim != 'D1':
            print('Requested simulation is not in the list of hosted simulations (M1, M2, D1, or D2). No known', 
                  'parameters on hand. Proceed at your own risk.',
                  'Numbers of particles will be computed on-the-fly. A pointer path must be specified')
            self.pointfpath = None
        
            #initial values as determined by BN-2 simulation
            # there are 219,607,640 star particles in the disk, 21,960,680 in the bulge, 
            #and 153,599 in the dwarf satellite -- aka a total of 241,721,919 stars
            # there are 878431120 dark particles in big galaxy, 2880684 in the satellite
            self.ntot = None
            self.ndark= None
            self.nstar= None
            self.masscutoff = 400 #mass flag boundary between bulge and disk 
        
        #set a string to be appended into save formats
        self.savestr=''

        #simulation data output info
        self.ncores = None #number of cores the particles are spread between, M1 is 24, D2 is 32
        

        #params for functions
        self.targetangle = 0 #for the bar frame adjustment target
        self.downsamplefrac = 0.00005 #fraction to downsample to
        self.downsamplen = None #if this is set it will take priority over the downsample frac option
        self.find_angle_rbounds = [1.5,2.5] #r bounds to compute the fft for determining the bar location, all in kpc
        self.find_angle_m = 2#looking for the m=2 mode


        #boolean loading type params
        self.wdm = 0 #toggle in dark matter or not
        self.barframe = 0 #default is to not load in the bar frame
        self.kmpers = 1 #set params as km/s, there's really no reason to change this ever in my opinion
        self.sort = 1 #sort in creating the pointers and assume that the pointers are sorted, you should never change this

        #default bounds for grabbing particles or snaps
        self.start = 0
        self.finish = None
        self.times = None #this will be made during the init

        #load the pattern speeds for the array
        self.patternspeeds = None 

        #data types
        self.infodtype = [('time','d'),('n','i'),('ndim','i'),('ng','i'),('nd','i'),('ns','i'),('on','i')]
        self.stellardtype = [('mass','f'), ('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('metals','f'), ('tform','f'), ('ID','Q')]
        self.dmdtype = [('mass','f'), ('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('ID','Q')]

        #data types for saving any snap or particle
        #self.commondtype = [('t','d'),('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('vr','f'), ('vphi','f'), ('vzz','f'), ('r','f'), ('phi','f'), ('zz','f'), ('mass','f'), ('idd','Q')]
        
        self.commondtype = [('t','d'),('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'),  ('mass','f'), ('idd','Q')]
        
        #this one should never be accessed, there are a couple old files though that may be in this format -- if needed though it's here
        self.oldsnapdtype = [('t','d'), ('idd','Q'),('x','f'), ('y','f'), ('z','f'), ('vx','f'), ('vy','f'), ('vz','f'), ('vr','f'), ('vphi','f'), ('vzz','f'), ('r','f'), ('phi','f'), ('zz','f'), ('mass','f')]


        #data type for the pointer files that particle tracing functions use
        self.pointdtype = [('idd','Q'),('core','H'), ('db1', 'H'),('db2', 'H'), ('seek','I')] ####need to add in which DB!
        self.errrange = 200 #default range to look on either side of where something /should/ be in a pointer file
        self.approxhalofrac = .1 #approximage halo fraction of the stellar population to speed up pointer file use

        #if not(os.path.isdir('plots')):
        #    print('Making plots/ directory for output figures.', flush=True)
         #   os.path.mkdir('plots')

        print('end at: '+str(datetime.datetime.now()),flush=True)

    def gettimes(self, return_array=False):
        path_01 = self.path + 'data01_01' + '/' + str(self.sim) + '/'
        if self.times is None:
            try:
                self.times = sorted(glob.glob(path_01+'snapshot*-0')) #remember! glob is not sorted!
                for ii,t in enumerate(self.times):
                    self.times[ii] = t.split('/')[-1]
            except:
                self.times = 'snapshot__00000.0000-0'
        
        if return_array == True:
            return self.times
                
    def timestot(self,s):
        return(float(s.split('-')[-2].split('_')[-1])*9.778145/1000.) #Gyr 

    def loadpatternspeeds(self):
        #try to load the pattern speed array for this sim if it's available
        if self.patternspeeds is None:
            try:
                self.patternspeeds = np.load(self.pointfpath+'patternspeed.npy')
            except:
                print('ATTN: Cannot load pattern speeds because file does not exist for this simulation at %s'%self.path, flush=True)

    #%%
    def printnow(self):
        print('check: '+str(datetime.datetime.now()),flush=True)

    #loaders
    #% load one snapshot
    def loader(self,filename,infoonly=0,forcedarkmatter=None):
        """
        loads an individual time snapshot from an individual node 
        converts to physical units

        Inputs
        ------------------
        filename (str): path to file of snapshot
        wdm (boolean): with dark matter toggle, will get stars and dark matter

        Returns
        ------------------
        cats (np.ndarray): stars [mass value for disk or bluge (float32), x position(float32),
                                y position (float32), z position (float32), vx x velocity(float32),
                                vy y velocity(float32), vz z velocity(float32), metals (float32),
                                tform (float), ID (uint64)]
        if self.wdm==True: 
            returns tuple with (catd,cats,info) where catd (np.ndarray): dark matter particles [same as above]
        """
        catd = []
        cats = []
        with open(filename, 'rb') as f:
            if infoonly == True:
                if self.verbose>1:
                    print(filename)
                #file info
                info= np.fromfile(f,dtype=self.infodtype,count=1)
                infoBytes = f.tell()
                if self.verbose>2:
                    print(infoBytes)
            elif self.wdm == False:
                if self.verbose>1:
                    print(filename)
                #file info
                info= np.fromfile(f,dtype=self.infodtype,count=1)
                infoBytes = f.tell()
                if self.verbose>2:
                    print(infoBytes)
                #skip darkmatter
                #read the first dm line
                if self.verbose>2:
                    print(f.tell())
                catd = np.fromfile(f,dtype=self.dmdtype, count=1)   
                #get the bytes location and subtract off the bytes location after loading info to get n bytes a line for dm
                if self.verbose>2:
                    print(f.tell())
                current = f.tell()
                dmBytes = current-infoBytes
                f.seek(dmBytes*(info['nd'][0]-1)+current)
                if self.verbose>2:
                    print(f.tell())
                # stars setup                    
                cats= np.fromfile(f,dtype=self.stellardtype, count=info['ns'][0])
                if self.verbose>2:
                    print('done')
            else:
                if self.verbose>1:
                    print(filename)
                #file info
                info= np.fromfile(f,dtype=self.infodtype,count=1)
                if self.verbose>2:
                    print(f.tell())
                #dark matter setup count is reading the number of rows
                catd= np.fromfile(f,self.dmdtype, count=info['nd'][0]) 
                if self.verbose>2:
                    print(f.tell())                   
                # stars setup                    
                cats= np.fromfile(f,dtype=self.stellardtype, count=info['ns'][0])
                if self.verbose>2:
                    print('done')
            
        if infoonly == False:
            #convert to physical units as found in README.md
            if self.wdm == True:
                catd['mass']*=2.324876e9
                if self.kmpers == 1:
                    catd['vx']*=100.
                    catd['vy']*=100.
                    catd['vz']*=100.
            cats['mass']*=2.324876e9
            if self.kmpers == True:
                    cats['vx']*=100.
                    cats['vy']*=100.
                    cats['vz']*=100.
        
        if (self.wdm == True) or (forcedarkmatter==True):
            return(catd,cats,info)
        else:
            return(cats,info)

    #% load one whole timestep, load a single snap
    def loadwholesnap(self,timestepid,forcebarframe=None,forcedarkmatter=None):
        
        #check that times exists
        self.gettimes()

        if self.kmpers != 1:
            print('ATTN: Enabled 100 km/s units instead of km/s for velocities.')
        direcs = os.listdir(self.path)
        c = 0
        coreid_array = np.array([])
        sim_direcs = np.array([])
        for i in direcs:
            if self.sim in os.listdir(self.path+str(i)):
                c+=1
                coreid = os.listdir(self.path+str(i)+'/'+str(self.sim))[0].split('-')[1]
                coreid_array = np.append(coreid_array, coreid)
                sim_direcs = np.append(sim_direcs, i)
        #if the number of total, dark, and stellar particles are not set then determine
        if self.ncores is None:
            self.ncores = c
        if self.ntot==None:
            if self.verbose >0:
                print('Loading to find nstars, ndark, and ntot'+self.path+str(sim_direcs[0])+self.times[timestepid][:-1]+str(coreid_array[0]), flush=True)
            _,info=self.loader(self.path+ str(sim_direcs[0]) + '/' + str(self.sim) + '/' + self.times[timestepid][:-1]+str(coreid_array[0]),infoonly=1)
            self.ntot=info['n']
            self.ndark=info['nd']
            self.nstar=info['ns']
            if self.ncores is None:
                self.ncores = c
            for j in range(1,self.ncores): #loader will just load the info line and leave
                coreid = coreid_array[j]
                sim_path = sim_direcs[j]
                print(sim_path)
                if self.verbose >1:
                    print('Loading '+self.path+str(sim_path)+self.times[timestepid][:-1]+str(coreid), flush=True)
                _,info=self.loader(self.path+str(sim_path)+'/'+str(self.sim)+'/'+self.times[timestepid][:-1]+str(coreid),infoonly=1)
                self.ntot=self.ntot+info['n']
                self.ndark=self.ndark+info['nd']
                self.nstar=self.nstar+info['ns']
                if self.verbose >2:
                    print('in length maker',self.ntot,self.ndark,self.nstar,info['n'],info['nd'],info['ns'],flush=True)

        #now load the first core for information and to start filling the array
        sim_path = sim_direcs[0] #first one
        coreid = coreid_array[0]
        if self.verbose > 1:
            print('\nLoading (also to set info) ', self.path+str(sim_path)+'/'+str(self.sim)+'/'+self.times[timestepid][:-1]+str(coreid), flush=True)
        if (self.wdm == True) or (forcedarkmatter==True):
            catd,cats,info=self.loader(self.path+str(sim_path)+'/'+str(self.sim)+'/'+self.times[timestepid][:-1]+str(coreid),forcedarkmatter=forcedarkmatter)

        else:
            cats,info=self.loader(self.path+str(sim_path)+'/'+str(self.sim)+'/'+self.times[timestepid][:-1]+str(coreid))

        if self.verbose > 1:
            print(info['n'],info['ns'],info['nd'], flush=True)

        if self.verbose > 2:
            print(self.ntot,self.ndark,self.nstar, flush=True) 

        #allocate for the snap arrays    
        snaparr = np.empty(self.nstar,dtype=self.commondtype)  
        if (self.wdm == True) or (forcedarkmatter==True):
            snaparr_dark = np.empty(self.ndark,dtype=self.commondtype)

        tnstar=int(info['ns'])
        if (self.wdm == True) or (forcedarkmatter==True):
            tndark=int(info['nd'])

        snaparr['x'][0:tnstar]=cats['x']
        snaparr['y'][0:tnstar]=cats['y']
        snaparr['z'][0:tnstar]=cats['z']
        snaparr['vx'][0:tnstar]=cats['vx']
        snaparr['vy'][0:tnstar]=cats['vy']
        snaparr['vz'][0:tnstar]=cats['vz']
        snaparr['mass'][0:tnstar]=cats['mass']
        snaparr['idd'][0:tnstar]=cats['ID']
        snaparr['t'][0:tnstar]=info['time']*9.778145/1000.

        arrayindx=tnstar

        if (self.wdm == True) or (forcedarkmatter==True):
            snaparr_dark['x'][0:tndark]=catd['x']
            snaparr_dark['y'][0:tndark]=catd['y']
            snaparr_dark['z'][0:tndark]=catd['z']
            snaparr_dark['vx'][0:tndark]=catd['vx']
            snaparr_dark['vy'][0:tndark]=catd['vy']
            snaparr_dark['vz'][0:tndark]=catd['vz']
            snaparr_dark['mass'][0:tndark]=catd['mass']
            snaparr_dark['idd'][0:tndark]=catd['ID']
            snaparr_dark['t'][0:tndark]=info['time']*9.778145/1000.

            arrayindx_dark = tndark


        if self.verbose > 2:
            print('stellar index is ',arrayindx, flush=True)
        for j in range(1,self.ncores):
            coreid = coreid_array[j]
            sim_path = sim_direcs[j]
            if self.verbose > 1:
                print('Loading '+self.path+str(sim_path)+'/'+str(self.sim)+'/'+self.times[timestepid][:-1]+str(coreid), flush=True)
            if (self.wdm == True) or (forcedarkmatter==True):
                catd,cats,info=self.loader(self.path+str(sim_path)+'/'+str(self.sim)+'/'+self.times[timestepid][:-1]+str(coreid),forcedarkmatter=forcedarkmatter)
            else:
                cats,info=self.loader(self.path+str(sim_path)+'/'+str(self.sim)+'/'+self.times[timestepid][:-1]+str(coreid))
            tnstar=int(info['ns'])
            if (self.wdm == True) or (forcedarkmatter==True):
                tndark = int(info['nd'])

            if self.verbose > 2:
                print(arrayindx,tnstar,arrayindx+tnstar, flush=True)
            snaparr['x'][arrayindx:arrayindx+tnstar]=cats['x']
            snaparr['y'][arrayindx:arrayindx+tnstar]=cats['y']
            snaparr['z'][arrayindx:arrayindx+tnstar]=cats['z']
            snaparr['vx'][arrayindx:arrayindx+tnstar]=cats['vx']
            snaparr['vy'][arrayindx:arrayindx+tnstar]=cats['vy']
            snaparr['vz'][arrayindx:arrayindx+tnstar]=cats['vz']
            snaparr['idd'][arrayindx:arrayindx+tnstar]=cats['ID']
            snaparr['mass'][arrayindx:arrayindx+tnstar]=cats['mass']
            snaparr['t'][arrayindx:arrayindx+tnstar]=info['time']*9.778145/1000.

            arrayindx=arrayindx+tnstar

            if (self.wdm == True) or (forcedarkmatter==True):
                snaparr_dark['x'][arrayindx_dark:arrayindx_dark+tndark]=catd['x'] 
                snaparr_dark['y'][arrayindx_dark:arrayindx_dark+tndark]=catd['y'] 
                snaparr_dark['z'][arrayindx_dark:arrayindx_dark+tndark]=catd['z'] 
                snaparr_dark['vx'][arrayindx_dark:arrayindx_dark+tndark]=catd['vx'] 
                snaparr_dark['vy'][arrayindx_dark:arrayindx_dark+tndark]=catd['vy'] 
                snaparr_dark['vz'][arrayindx_dark:arrayindx_dark+tndark]=catd['vz']
                snaparr_dark['idd'][arrayindx_dark:arrayindx_dark+tndark]=catd['ID']
                snaparr_dark['mass'][arrayindx_dark:arrayindx_dark+tndark]=catd['mass']
                snaparr_dark['t'][arrayindx_dark:arrayindx_dark+tndark]=info['time']*9.778145/1000.

                arrayindx_dark=arrayindx_dark+tndark

            #except:
            #    print(times[i]+'-'+str(j),'has no star particles', flush=True)
        if self.recenter == True:
            snaparr = self.apply_recen_snap(snaparr) 

        #move into bar frame if true
        if (self.barframe == True) or (forcebarframe==True):
            #check for path
            if os.path.exists(self.pointfpath+'patternspeed.npy'):
                snaparr = self.intobarframe(snaparr,frame=timestepid)
                if (self.wdm == True) or (forcedarkmatter==True):
                    snaparr_dark = self.intobarframe(snaparr_dark,frame=timestepid)
            else: #this is so inefficient never use it and intobarframe isnt set up for it -- okay but what about with the smaller sims?
                print('ATTN: skipping moivng into bar pattern frame because speeds not computed yet')
                # pspeed = getpatternspeed(path,timestepid,inputsnaparr=snaparr,verbose=verbose,withbarangle=1) #also in radperGyr
                # snaparr = intobarframe(snaparr,timestepid)
        if (self.wdm == True) or (forcedarkmatter==True):
            return(snaparr,snaparr_dark)
        else:
            return(snaparr)

    #% load whole pickle step
    def loadstep(self,timestepid):
        fstr = self.path+'step'+str(timestepid)+'.p'
        
        #load the pickle
        with open(fstr,'rb') as f:
            idd,x,y,z,vx,vy,vz,mass = pickle.load(f)
        
        return (idd,x,y,z,vx,vy,vz,mass)

    def makepointerf(self):
        #check that times exists
        self.gettimes()
        
        #ncores = len(glob.glob(self.path+'*'+self.times[0][:-1]+'*'))
        
        ######
        direcs = os.listdir(self.path)
        c = 0
        coreid_array = np.array([])
        sim_direcs = np.array([])
        for i in direcs:
            if self.sim in os.listdir(self.path+str(i)):
                c+=1
                coreid = os.listdir(self.path+str(i)+'/'+str(self.sim))[0].split('-')[1]
                coreid_array = np.append(coreid_array, coreid)
                sim_direcs = np.append(sim_direcs, i)
        #if the number of total, dark, and stellar particles are not set then determine
        if self.ncores is None:
            self.ncores = c
        #####
        if self.verbose >0:
            print('Loading to find nstars, ndark, and ntot'+self.path+self.times[0], flush=True)
        
        #_,info=self.loader(self.path+self.times[0])
        _,info=self.loader(self.path+ str(sim_direcs[0]) + '/' + 
                           str(self.sim) + '/' + 
                           self.times[0][:-1]+
                           str(coreid_array[0]))
        self.ntot=info['n']
        self.ndark=info['nd']
        self.nstar=info['ns']
        ncores = c
        for j in range(1,ncores):
            ####
            coreid = coreid_array[j]
            sim_path = sim_direcs[j]
            ###
            if self.verbose > 1:
                print('Loading '+self.path+str(sim_path)+
                      self.times[0][:-1]+str(coreid), flush=True)            
            #_,info=self.loader(self.path+self.times[0][:-1]+str(j))
            _,info=self.loader(self.path+str(sim_path)+'/'+
                               str(self.sim)+'/'+
                               self.times[0][:-1]+
                               str(coreid))
            self.ntot=self.ntot+info['n'] 
            self.ndark=self.ndark+info['nd']
            self.nstar=self.nstar+info['ns']
            if self.verbose > 1:
                print('in length maker',self.ntot,self.ndark,self.nstar,info['n'],info['nd'],info['ns'],flush=True)
 
        starti = len(glob.glob(self.pointfpath+'*stellar*'))
        for ts,tt in enumerate(self.times[starti: self.pointfend]):
            ###this is a good spot to change this to a max time step -
            #set file name for the timestep pointer file
            tsstr = tt.split('-')[:-1][0]
            #print('tsstr', tsstr)
            if self.verbose >3:
                print('tsstr', tsstr)
            tsname = 'stellar_'+self.path.split('/')[-2]+'_'+tsstr
            #print('tsname', tsname)
            #?sort cause this could be used to make it faster later, cause they're unique we dont really need to do this
            #initalize this file's stellar array
            starpointarr = np.empty(self.nstar,dtype=self.pointdtype)
            
            arrayindx = 0
            for nc in range(0,ncores):
                if self.verbose >2:
                    print('core '+str(nc), flush=True)   
                #load nth core of timestep/snapshot 
                sim_path = sim_direcs[nc]
                core_id = coreid_array[nc]
                print(sim_path, core_id)
                fname = self.path+str(sim_path)+'/'+ str(self.sim)+'/'+str(tsstr)+'-'+str(core_id)
                #+str(nc)
                print(fname)
                if self.verbose >1:
                    print('Loading '+fname, flush=True)
                
                with open(fname, 'rb') as f:
                    # get byte lengths 
                    info = np.fromfile(f,dtype=self.infodtype,count=1)
                    infoBytes = f.tell()
                    if self.verbose>2:
                        print('Info byte length is '+str(infoBytes), flush=True)

                    #read the first dm line
                    catd = np.fromfile(f,dtype= self.dmdtype, count=1)  
                    current = f.tell()
                    dmBytes = current-infoBytes
                    if self.verbose>2:
                        print('DM byte length is '+str(dmBytes), flush=True)

                    #seek forward past rest of dm data
                    f.seek(dmBytes*(info['nd'][0]-1)+current)
                    if self.verbose>2:
                        print('Advanced through DM to '+str(f.tell()), flush=True)
                    
                    #get star byte info
                    cats= np.fromfile(f,dtype=self.stellardtype, count=1)
                    current = f.tell()
                    print('cats', cats, cats.shape)
                    if self.verbose >2:
                        print(f.tell())
                    stBytes = current-dmBytes*(info['nd'][0])-infoBytes
                    if self.verbose>2:
                        print('Stellar byte length is '+str(stBytes), flush=True)
                        print(f.tell())
                    #seek back to start of stars
                    startofStellar = current-stBytes
                    f.seek(startofStellar) 
                    if self.verbose>2:
                        print('Back to start of star '+str(f.tell()))
                    
                    # stars setup
                    if self.verbose >1:
                        print('loading stellar')
                        self.printnow()          
                    cats= np.fromfile(f,dtype=self.stellardtype, count=info['ns'][0])
                    afterStellar = f.tell()
                    print('after stars '+str(f.tell()))
                    print('info ns 0', info['ns'][0])
                    if info['ns'][0] > 0:
                        sim_path = str(sim_direcs[nc])
                        first_part = sim_path.split('data')[1][0:2]
                        second_part = sim_path.split('data')[1][3:5]
                        core_id = coreid_array[nc]
                        print('saving out idd, core, sim_direc, seek')
                        #print(core_id, sim_path, core_id.shape, 'simdirecs shape',
                         #    sim_direcs[nc].shape)
                        starpointarr['idd'][arrayindx:arrayindx+info['ns'][0]] = cats['ID']
                        starpointarr['core'][arrayindx:arrayindx+info['ns'][0]] = core_id
                        #breaking sim_path into first number, second number & saving out
                        starpointarr['db1'][arrayindx:arrayindx+info['ns'][0]] = int(first_part)
                        starpointarr['db2'][arrayindx:arrayindx+info['ns'][0]] = int(second_part)
                        #
                        starpointarr['seek'][arrayindx:arrayindx+info['ns'][0]] = np.arange(startofStellar,afterStellar,stBytes)
                        print('sim_path[0], first part, second part', sim_path, first_part, int(first_part),
                              second_part, int(second_part))
                    else:
                        if self.verbose>1:
                            print('No stellar particles in '+fname, flush=True)
                    if self.verbose >1:
                        self.printnow()
                    print(arrayindx)
                    arrayindx += info['ns'][0]
                    print(arrayindx)

            if self.verbose >1:
                print('saving timestep '+tsname)
                self.printnow()     
            
            #sort by idx so that the loc is the idd
            if self.sort == 1: #this is an integral part of this function and should never be skipped unless the sim is very small
                sortorder = np.argsort(starpointarr['idd'])

                sortpointer = np.empty(len(starpointarr),dtype=self.pointdtype)

                sortpointer['idd'] = starpointarr['idd'][sortorder]
                sortpointer['core'] = starpointarr['core'][sortorder]
                ####added
                sortpointer['db1']  = starpointarr['db1'][sortorder]
                sortpointer['db2']  = starpointarr['db2'][sortorder]
                sortpointer['seek'] = starpointarr['seek'][sortorder]

                #save as a pickle .npy file cause it's wicked fast
                
                print('up to save step!!')
                np.save(self.pointfpath+tsname,sortpointer)
            else:
                print('up to save step!!')
                #save as a pickle .npy file cause it's wicked fast
                np.save(self.pointfpath+tsname,starpointarr)
            
            if self.verbose >1:
                print('finished saving timestep '+tsname)
                self.printnow()  

    def downsamplesnap(self,snaparr,downsampmult=None):
        startinginds = range(len(snaparr))
        if self.downsamplen is None:
            if downsampmult is None:
                downsampleinds = np.random.choice(startinginds,
                                int(round(self.downsamplefrac*len(startinginds))),
                                replace=False)
            else:
                downsampleinds = np.random.choice(startinginds,
                                int(round(self.downsamplefrac*downsampmult*len(startinginds))),
                                replace=False)
        else:
            if downsampmult is None:
                downsampleinds = np.random.choice(startinginds,int(self.downsamplen),replace=False)
            else:
                downsampleinds = np.random.choice(startinginds,int(self.downsamplen*downsampmult),replace=False)

        return(snaparr[downsampleinds])

    #%% load one particle over time
    def loadonesource(self,idx,start=None,finish=None,forcebarframe=0):
        '''
        takes a path to the folder containing snapshots and an id
        returns an array 
            dtype=[('t','f'),('x','f'), ('y','f'), ('z','f'), 
                ('vx','f'), ('vy','f'), ('vz','f'),('mass','f'), ('idd','Q')] 

        set 'npypointerpath' to pointfpath
        or 'loadall': loadall will toggle to itterate through and load all the arrays
                                        npypointer will access .npy pointer files

        '''
        if self.kmpers != 1:
            print('ATTN: Enabled 100 km/s units instead of km/s for velocities.')
        idx = np.uint64(idx) #force this datatype 

        #get start and finish from params if none, allows for itterative loading
        if start is None:
            start = self.start
        if finish is None:
            finish = self.finish

        #check for times
        self.gettimes()
        ##fixed this
        #if self.ncores == None:
         #   ncores = len(glob.glob(self.path+'*'+self.times[0][:-1]+'*'))
        direcs = os.listdir(self.path)
        c = 0
        coreid_array = np.array([])
        sim_direcs = np.array([])
        for i in direcs:
            if self.sim in os.listdir(self.path+str(i)):
                c+=1
                coreid = os.listdir(self.path+str(i)+'/'+str(self.sim))[0].split('-')[1]
                coreid_array = np.append(coreid_array, coreid)
                sim_direcs = np.append(sim_direcs, i)
        #if the number of total, dark, and stellar particles are not set then determine
        if self.ncores is None:
            self.ncores = c
        #set other params
        if (finish == None) or (finish > len(self.times)):
            print('ATTN: Setting finish tind as ', str(len(self.times)))
            finish = len(self.times)

        # let's allocate space for the finished final array for one source
        tlen = len(range(start,finish))
        sourcearr=np.empty(tlen,dtype=self.commondtype)

        if self.pointfpath == None:
            indvarrayindx=0
            lastcore = 0

            for i in range(start,finish): 
                #looping over times -
                keepgoing = 1
                if keepgoing > 0:
                    if self.verbose > 1:
                        #print('Loading starting from last core first '+self.path+self.times[i][:-1]+str(lastcore), flush=True)
                        print('loading ', self.path+ str(sim_direcs[0]) + '/' + str(self.sim) + '/' +  self.times[i][:-1]+ str(coreid_array[0]))
                    if self.wdm == True:
                        _,cats, info=self.loader(self.path+ str(sim_direcs[0]) + '/' + str(self.sim) + '/' + self.times[i][:-1]+ str(coreid_array[0]))
                        # _,cats,info=   self.loader(self.path+self.times[i][:-1]+str(lastcore))
                    else:
                        cats,info= self.loader(self.path+ str(sim_direcs[0]) + '/' + str(self.sim) + '/' + self.times[i][:-1]+str(coreid_array[0]))
                        #self.loader(self.path+self.times[i][:-1]+str(lastcore))
                    if np.where(np.isin(cats['ID'],idx,assume_unique=1))[0].size > 0:
                        j = lastcore
                        if self.verbose > 0:
                            print('Source found in '+self.path+self.times[i][:-1]+str(j), flush=True)
                        lastcore = j
                        if self.verbose > 1:
                            print('Last core is ',j)
                        foundid = np.argwhere(idx == cats['ID']).flatten()
                        if self.verbose >1:
                            print('printing cats[foundid]')
                            print(cats[foundid])

                        sourcearr['x'][indvarrayindx]=cats['x'][foundid]
                        sourcearr['y'][indvarrayindx]=cats['y'][foundid]
                        sourcearr['z'][indvarrayindx]=cats['z'][foundid]
                        sourcearr['vx'][indvarrayindx]=cats['vx'][foundid]
                        sourcearr['vy'][indvarrayindx]=cats['vy'][foundid]
                        sourcearr['vz'][indvarrayindx]=cats['vz'][foundid]
                        sourcearr['mass'][indvarrayindx]=cats['mass'][foundid]
                        sourcearr['idd'][indvarrayindx]=cats['ID'][foundid]


                        #sourcearr['vr'][indvarrayindx],sourcearr['vphi'][indvarrayindx],sourcearr['vzz'][indvarrayindx]=coords.rect_to_cyl_vec(cats['vx'][foundid],cats['vy'][foundid],cats['vz'][foundid],cats['x'][foundid],cats['y'][foundid],cats['z'][foundid])
                        #sourcearr['r'][indvarrayindx],sourcearr['phi'][indvarrayindx],sourcearr['zz'][indvarrayindx]=coords.rect_to_cyl(cats['x'][foundid],cats['y'][foundid],cats['z'][foundid])
                        sourcearr['t'][indvarrayindx]=info['time']*9.778145/1000.

                        keepgoing = 0

                if keepgoing > 0:
                    for j in range(0,ncores):
                        if keepgoing > 0:
                            if self.verbose > 1:
                                print('Loading '+self.path+self.times[i][:-1]+str(j), flush=True)
                                print('actually loading', self.path+ str(sim_direcs[j]) + '/' + str(self.sim) + '/' +  self.times[i][:-1]+str(coreid_array[j]))
                            if self.wdm == True:
                                _,cats,info= self.loader(self.path+ str(sim_direcs[j]) + '/' +  str(self.sim) + '/' +  self.times[i][:-1]+  str(coreid_array[j]))
                                #self.loader(self.path+self.times[i][:-1]+str(j))
                            else:
                                cats,info= self.loader(self.path+ str(sim_direcs[j]) + '/' + str(self.sim) + '/' +  self.times[i][:-1]+ str(coreid_array[j]))
                                #self.loader(self.path+self.times[i][:-1]+str(j))
                            if np.where(np.isin(cats['ID'],idx,assume_unique=1))[0].size > 0:
                                if self.verbose > 0:
                                    #print('Source found in '+self.path+self.times[i][:-1]+str(j), flush=True)
                                    print('Source found in' + self.path+ str(sim_direcs[j]) + '/' + str(self.sim) + '/' +  self.times[i][:-1]+str(coreid_array[j]))
                                lastcore = j
                                if self.verbose > 1:
                                    print('Last core is ',j)
                                foundid = np.argwhere(idx == cats['ID']).flatten()

                                if self.verbose >1:
                                    print('printing cats[\'x\'][foundid]')
                                    print(cats['x'][foundid])

                                if self.verbose >1:
                                    print('printing cats[\'ID\'][foundid]')
                                    print(cats['ID'][foundid])
                                sourcearr['x'][indvarrayindx]=cats['x'][foundid]
                                sourcearr['y'][indvarrayindx]=cats['y'][foundid]
                                sourcearr['z'][indvarrayindx]=cats['z'][foundid]
                                sourcearr['vx'][indvarrayindx]=cats['vx'][foundid]
                                sourcearr['vy'][indvarrayindx]=cats['vy'][foundid]
                                sourcearr['vz'][indvarrayindx]=cats['vz'][foundid]
                                sourcearr['mass'][indvarrayindx]=cats['mass'][foundid]
                                sourcearr['idd'][indvarrayindx]=cats['ID'][foundid]


                                #sourcearr['vr'][indvarrayindx],sourcearr['vphi'][indvarrayindx],sourcearr['vzz'][indvarrayindx]=coords.rect_to_cyl_vec(cats['vx'][foundid],cats['vy'][foundid],cats['vz'][foundid],cats['x'][foundid],cats['y'][foundid],cats['z'][foundid])
                                #sourcearr['r'][indvarrayindx],sourcearr['phi'][indvarrayindx],sourcearr['zz'][indvarrayindx]=coords.rect_to_cyl(cats['x'][foundid],cats['y'][foundid],cats['z'][foundid])
                                sourcearr['t'][indvarrayindx]=info['time']*9.778145/1000.

                                keepgoing = 0
                indvarrayindx+=1
                #error out if cannot find the ID
                if keepgoing > 0:
                    print('ID NOT FOUND in '+self.path+self.times[i][:-1])
                    break
        else:
            #check paths error 
            print('found pointer path')
            if os.path.exists(self.pointfpath):
                indvarrayindx=0
                lspoint = os.listdir(self.pointfpath) 
                lspoint.sort(key=lambda s:s.split('_')[-1].split('.npy')[0])
                #assumes that if patternspeed.npy is there, it will be the last file after sorting, could compare first and last item if this isnt always true
                if lspoint[-1].find('patternspeed')>-1:
                    lspoint = lspoint[:-1]
                for i in range(start,finish): #looping through time
                    pointerpath = self.pointfpath+lspoint[i]
                    #get pointer file
                    if self.verbose >2:
                        print('load pointer path: ',pointerpath)
                        self.printnow()
                    pointer = np.load(pointerpath,mmap_mode='r',allow_pickle=1) 

                    if self.sort == 0:
                        bf = datetime.datetime.now()
                        found = pointer[pointer['idd']==idx]
                        if self.verbose > 2:
                            print('time for starpointarr[idx] is ',datetime.datetime.now()-bf)
                            print('found under sort', found)
                    else:
                        #offset application
                        if idx >10000000000000:
                            print('large idx!')
                            if self.verbose > 2:
                                bf = datetime.datetime.now()
                            try:
                                #grab sub array of region with halo particles 
                                subr = pointer[-round(len(pointer)*self.approxhalofrac+self.errrange):]

                                found = subr[subr['idd']==idx]
                                if self.verbose >2:
                                    print('time for subr[subr[\'idd\']==idx] is ',datetime.datetime.now()-bf)
                            except:
                                print('\n\nISSUE WITH FILE: ',pointerpath)
                                print('pointer[0]: ',pointer[0])
                                print('id is ', idx)
                                if pointer['idd'][0]==pointer['seek'][0]:
                                    print('all zeros')


                        else:
                            if self.verbose > 2:
                                bf = datetime.datetime.now()
                            found=pointer[[idx]]
                            if self.verbose > 2:
                                print('found', found)
                            #catch when there's this weird offset of ~200 skipped IDs
                            if pointer[[idx]]['idd'][0]!=idx:
                                if self.verbose >1:
                                    print('using error range to find local source')
                                #try except for end of array 
                                try:
                                    subarr = pointer[int(idx-self.errrange):int(idx+self.errrange)]
                                except:
                                    subarr = pointer[int(idx-self.errrange):]
                                found = subarr[subarr['idd']==idx]
                                #print('found 2 ', found)
                                if len(found) <1:
                                    if self.verbose >1:
                                        print('expanding error range to find local source')
                                    self.errrange *=2
                                    try:
                                        subarr = pointer[int(idx-self.errrange):int(idx+self.errrange)]
                                    except:
                                        subarr = pointer[int(idx-self.errrange):]
                                    found = subarr[subarr['idd']==idx]
                                    if len(found) <1:
                                        print('\n\nthere\'s something wrong, error range expansion isn\'t enough to solve it')

                            if self.verbose > 2:
                                print('time for pointer[idx] is ',datetime.datetime.now()-bf)

                    #load the direct file
                    try:
                        if self.verbose > 0:
                            print('printing found under try', found)
                            print('Loading '+pointerpath, flush=True)
                            if self.verbose >2:
                                self.printnow()
                        #getting from DB1, DB2 to sim directory - 
                        db1 = found['db1'][0]
                        db2 = found['db2'][0]
                        #if db1 is < 10, add zero in front of db1 bc file format is e.g. data05_02, data10_02
                        #db2 is always between 01 and 03
                        if db1 <= 9:
                            sim_loc = 'data0' + str(db1) + '_' + '0' + str(db2)
                        if db1 > 9:
                            sim_loc = 'data' + str(db1) + '_' + '0' + str(db2)
                        if db2 > 3:
                            print('uhoh, db2 is bigger than 3!')
                        filename = self.path+ str(sim_loc) + '/' + str(self.sim) + '/' + self.times[i][:-1]+ str(found['core'][0])
                        #self.path+ str(found['sim_direc'][0]) + '/' + str(self.sim) + '/' + self.times[i][:-1]+ str(found['core'][0])
                        #self.path+ str(sim_direcs[j]) + '/' + str(self.sim) + '/' + self.times[i][:-1]+ str(coreid_array[j])
                        #self.path+self.times[i][:-1]+str(found['core'][0])
                        
                        if self.verbose>3:
                            print('filename', filename)
                            print('direc is', db1, db2, sim_loc)
                            print('core is ',found['core'][0])
                        with open(filename, 'rb') as f:
                            info= np.fromfile(f,dtype=self.infodtype,count=1)
                            if self.verbose>3:
                                print('seek to ',found['seek'][0])
                            f.seek(found['seek'][0])
                            cats= np.fromfile(f,dtype=self.stellardtype, count=1)

                            if self.verbose >2:
                                print('insert into indvarray')
                                self.printnow()
                            if self.verbose >1:
                                bf = datetime.datetime.now()
                            sourcearr['x'][indvarrayindx]=cats['x']
                            sourcearr['y'][indvarrayindx]=cats['y']
                            sourcearr['z'][indvarrayindx]=cats['z']
                            if self.kmpers == 1:
                                    cats['vx']*=100.
                                    cats['vy']*=100.
                                    cats['vz']*=100.
                            sourcearr['vx'][indvarrayindx]=cats['vx']
                            sourcearr['vy'][indvarrayindx]=cats['vy']
                            sourcearr['vz'][indvarrayindx]=cats['vz']
                            sourcearr['mass'][indvarrayindx]=cats['mass'] * 2.324876e9 #converting to msun
                            sourcearr['idd'][indvarrayindx]=cats['ID']
                            if self.verbose >1:
                                print('time for plugging into array is ',datetime.datetime.now()-bf)
                            if self.verbose >2:
                                print('coords transf')
                                self.printnow()
                            if self.verbose >1:
                                bf = datetime.datetime.now()
                            #sourcearr['vr'][indvarrayindx],sourcearr['vphi'][indvarrayindx],sourcearr['vzz'][indvarrayindx]=coords.rect_to_cyl_vec(cats['vx'],cats['vy'],cats['vz'],cats['x'],cats['y'],cats['z'])
                            #sourcearr['r'][indvarrayindx],sourcearr['phi'][indvarrayindx],sourcearr['zz'][indvarrayindx]=coords.rect_to_cyl(cats['x'],cats['y'],cats['z'])
                            sourcearr['t'][indvarrayindx]=info['time']*9.778145/1000.
                            if self.verbose >1:
                                print('time for computing and plugging new vals into array is ',datetime.datetime.now()-bf)
                            if self.verbose >2:
                                print('onto the next loop')
                                self.printnow()
                    except:
                        print('\n\nISSUE WITH FILE: ',pointerpath,flush=True)
                        print('found[0]: ',found[0],flush=True)
                    indvarrayindx+=1

        #move into bar frame if true
        if (self.barframe == True) or (forcebarframe == True):
            #check for path
            if os.path.exists(self.pointfpath+'patternspeed.npy'):
                sourcearr = self.intobarframe(sourcearr,start=start,finish=finish)
            else: #this is so inefficient never use it and intobarframe isnt set up for it
                print('ATTN: skipping moivng into bar pattern frame because speeds not computed yet', flush=True)
        return(sourcearr)



    def getperiapos(self,orbitarr,params=(['r','z']),returnperis=False,returnorbit=0,save=0,pathstr='',savestr=''):
        #for an individual orbit, get the peri- and apo-centers in each direction of params
        apos = []
        if returnperis == True:
            peris = []
        if self.barframe == False and (params.count('x')+params.count('y'))>0:
            print('ATTN: Cannot move forward with x or y as a param unless fixed in bar frame.', flush=True) #if in bar frame should be fine? -- check this
            exit
        #set data type to return the times and location of the parameter selected's peri/apo
        for pp in params: 
            #for r, compute r 
            if pp == 'r':
                param = np.sqrt(orbitarr['x']**2+orbitarr['y']**2)
            else:
                param = orbitarr[pp]

            #set the datatype
            periapodtype = [('t','d'),(str(pp),'d')]
            whereapos = find_peaks(param)[0]
            aposarr = np.empty(len(whereapos),dtype=periapodtype)
            
            #get the peaks and related times
            aposarr['t'] = orbitarr['t'][whereapos]
            aposarr[pp] = param[whereapos]

            #get peris or if doing z, must get peris to check if need to flip those for apos
            if returnperis == True or pp=='z':
                periapodtype = [('t','d'),(str(pp),'d')]
                whereperis = find_peaks(-param)[0]
                perisarr = np.empty(len(whereperis),dtype=periapodtype)
                perisarr['t'] = orbitarr['t'][whereperis]
                perisarr[pp] = param[whereperis]
                #check to see if the apocenter is in fact negative
                if np.abs(np.median(perisarr[pp]))>np.abs(np.median(aposarr[pp])):
                    apos.append(np.array(perisarr,dtype=perisarr.dtype))
                    if returnperis == True:
                        peris.append(np.array(aposarr,dtype=aposarr.dtype))
                else:
                    apos.append(np.array(aposarr,dtype=aposarr.dtype))
                    if returnperis == True:
                        peris.append(np.array(perisarr,dtype=perisarr.dtype))
            else:
                apos.append(np.array(aposarr,dtype=aposarr.dtype))
        
        if returnperis == True:
            if save == True:
                for pp in params:
                    np.save('%sparticle_%s_%s_peris_%s'%(pathstr,str(orbitarr['idd'][0]),pp,savestr),peris)
                    np.save('%sparticle_%s_%s_apos_%s'%(pathstr,str(orbitarr['idd'][0]),pp,savestr),apos)
            if returnorbit == True:
                return (peris,apos,orbitarr)
            else:
                return (peris,apos)
        else:
            if save == True:
                for ii,pp in enumerate(params):
                    np.save('%sparticle_%s_%s_apos_%s'%(pathstr,str(orbitarr['idd'][0]),pp,savestr),apos[ii])
            if returnorbit == True:
                return (apos,orbitarr)
            else:
                return apos

    # move into the bar frame
    def find_angle(self, snaparr, degBool=False):
        '''
        to here !
        adjusted to compute r, phi, z on-the-fly, no longer automatically computed
        for all stars!
        Computes the angle of the m[=2] bar to the 0 phi axis

        Inputs
        ---------------
        snaparr (array) [dtype: commondtype]: must be loaded with loadwholesnap()
        degBool (bool): default is False to return angle in radians
        rmin,rmax (floats) = radial bounds for selecting the region of the disk to compute the fft on
        m=2 mode as the bar mode

        Returns
        ---------------
        barangle (float): returns a float value in radians of the bar angle compared to snaparr['phi']=zero
        '''
        #snaparr = self.recen_wholesnap(snaparr) ###added recentering
        rmin = self.find_angle_rbounds[0]
        rmax = self.find_angle_rbounds[1]
        disk_flag = (snaparr['mass']<self.masscutoff)
        #rr = np.sqrt(snaparr['x'][disk_flag]**2+snaparr['y'][disk_flag]**2)
        r, phi, zz =coords.rect_to_cyl(snaparr['x'][disk_flag],snaparr['y'][disk_flag],snaparr['z'][disk_flag])
        barsample = (r<rmax)&(r>rmin)
        #binning from -pi to pi in 360 steps, i.e. degree at a time
        counts, _ = np.histogram(phi[barsample], bins = np.linspace(-np.pi, np.pi, 360))
        
        #fourier transform the counts removing the offset
        ff=np.fft.fft(counts-np.mean(counts)) 

        barangle = -np.angle(ff[self.find_angle_m],deg=degBool)/self.find_angle_m
        return barangle
    ###
    ### we're figuring out how to move into the bar frame when loading a single snap or a single source, oh are we lol
    ###
    def computepatternspeed(self,start=None,finish=None):
        '''
        Create a csv file with the pattern speed in rad/Gyr and km/s over the time bounds provided. 
        rmin,rmax set the distance to compute the speed at and then their average value is the location for km/s pattern speed tangental
        '''
        
        #check for times
        self.gettimes()

        #get start and finish
        if start is None:
            start = self.start
        if finish is None:
            finish = self.finish

        if finish is None:
            finish = len(self.times)
            
        outarr = np.empty((finish-start,1),dtype=[('t','d'),('tind','H'),('barangle','f'),('patternspeed_radperGyr','f'),('patternspeed','f')])

        #compute bar angle for step ahead if not first timestep
        if start == 0:
            old_bar_angle = 0
        else:
            if self.verbose >0:
                print('Loading ',self.path)
            pastsnaparr = self.loadwholesnap(start-1)
            pastsnaparr = self.apply_recen_snap(pastsnaparr)#self.recen_wholesnap(pastsnaparr) #recentering ---
            old_bar_angle = self.find_angle(pastsnaparr)

        for i, tt in enumerate(range(start,finish)):
            if self.verbose >0:
                print('Loading ',self.path)
            snaparr = self.loadwholesnap(tt)
            snaparr = self.apply_recen_snap(snaparr) #self.recen_wholesnap(snaparr) #recentering ---
            #if first loop then set physical params
            if tt == start:
                if self.verbose >1:
                    print('First loop in time range, determining timestep with particle ',str(snaparr['idd'][0]), 'between ',str(self.times[start+1]),str(self.times[start+2]))
                #grab the next timestep to measure the time resolution
                timestep = self.loadonesource(snaparr['idd'][0],start=start+1,finish=start+2)['t'][0]-snaparr['t'][0]
            
            #for current step, get the bar angle and pattern speed 
            bar_angle = self.find_angle(snaparr)
            pattern_speed = (old_bar_angle-bar_angle)/(timestep)

            #bar_angle > old_bar_angle, if difference is near 180, flip it
            if abs(old_bar_angle-bar_angle)>=(np.pi*3/4):
                pattern_speed = (old_bar_angle - (bar_angle + np.pi))/(timestep)
            
            if self.verbose >1:
                print('Bar angle is ', bar_angle, 'radians', flush=True)
                print('Pattern speed is ', pattern_speed, 'radians per Gyr', flush=True)

            
            outarr[i]['t'] = snaparr['t'][0]
            outarr[i]['tind'] = tt
            outarr[i]['barangle'] = bar_angle
            outarr[i]['patternspeed_radperGyr'] = pattern_speed
            outarr[i]['patternspeed'] = pattern_speed*(u.radian/u.Gyr*u.kpc).to(u.km/u.s,equivalencies=u.dimensionless_angles()) #conversion factor to go from rad/Gyr to km/s/kpc 
            
            self.printnow()
            if self.verbose >0:
                print('now at ', outarr[tt]['t'],flush=True)
            old_bar_angle = bar_angle
            np.save(self.pointfpath+'patternspeed',outarr)
        
        #save as a .npy in the pointer file path
        np.save(self.pointfpath+'patternspeed',outarr)

        # #put into a dataframe to save
        # bar_info = pd.DataFrame(np.array([outarr['t'],outarr['barangle'], outarr['patternspeed_radperGyr'], outarr['patternspeed']]).T, 
        #                     columns=['time','bar_angle','pattern_speed_radperGyr', 'pattern_speed_kmpersperkpc'])
        # bar_info.to_csv('bar_info.csv')
        
        return outarr

    def getpatternspeed(self,tidx,withbarangle=True,prior_bar_angle=None,timestep=None,inputsnaparr=None):
        '''
        Function to compute the pattern speed [in Rad/Gyr] at any given snap. Will always return the patternspeed and 
        can opt to include bar angles of target snap to facilitate computation over time without reloading redundantly.

        If itterating over in series, first loop do not set prior_bar_angle but do return withbarangle,
            then go into loop with prior_bar_angles from previous step.
        
        If computing for an array that is already loaded, must still provide the time index in tidx (it checks to see they match),
            but then it can skip loadwholesnap of the targetarray though it will still have to load the prior one.
        '''
        #get times set
        self.gettimes()

        #if there is a pattern speed file for the simulation, grab from it 
        if os.path.exists(self.pointfpath+'patternspeed.npy'):
            patternspeeds = np.load(self.pointfpath+'patternspeed.npy',mmap_mode='r+',allow_pickle=1)
            patterninfo = patternspeeds[['tind']==tidx]
            bar_angle = patterninfo['barangle']
            timestep = patternspeeds[['tind']==tidx-1]-patterninfo['t']
            pattern_speed = patterninfo['patternspeed_radperGyr']
        
        #assume that you will load the desired timestep targeted whole snap and the snap prior unless they're provided
        else:
            #check for input array
            if inputsnaparr is not None:
                #check that the times match
                if round(inputsnaparr['t'][0],3) == round(self.timestot(self.times[tidx]),3):
                    inputsnaparr = self.apply_recen_snap(inputsnaparr)#self.recen_wholesnap(inputsnaparr) #recentering ---
                    bar_angle = self.find_angle(inputsnaparr)
                else:
                    print('ATTN: time index is ', tidx, 'for times[tidx]', self.times[tidx],'while input timearr is at t=',str(inputsnaparr['t'][0]))
                    print('Running loadwholesnap() at provided tidx.')
                    targetsnaparr = self.loadwholesnap(tidx)
                    targetsnaparr = self.apply_recen_snap(targetsnaparr)#self.recen_wholesnap(targetsnaparr) #recentering ---
                    bar_angle = self.find_angle(targetsnaparr)

            else:
                #load target info
                targetsnaparr = self.loadwholesnap(tidx)
                targetsnaparr = self.apply_recen_snap(targetsnaparr)#self.recen_wholesnap(targetsnaparr) #recentering ---
                bar_angle = self.find_angle(targetsnaparr)
            
            #check for input prior bar angle
            if prior_bar_angle is not None:
                old_bar_angle = prior_bar_angle
                timestep = timestep
            else:
                priorsnaparr = self.loadwholesnap(tidx-1)
                priorsnaparr = self.apply_recen_snap(priorsnaparr)#self.recen_wholesnap(priorsnaparr) #recentering ---
                old_bar_angle = self.find_angle(priorsnaparr)
                #compute timestep
                timestep = targetsnaparr['t'][0]-priorsnaparr['t'][0]

            #bar_angle > old_bar_angle, if difference is near 180, flip it
            pattern_speed = (old_bar_angle-bar_angle)/(timestep)
            if abs(old_bar_angle-bar_angle)>=(np.pi*3/4):
                pattern_speed = (old_bar_angle - (bar_angle + np.pi))/(timestep)
            
            if self.verbose >1:
                print('Bar angle is ', bar_angle, 'radians', flush=True)
                print('Pattern speed is ', pattern_speed, 'radians per Gyr',flush=True)

        if withbarangle == True:
            return (bar_angle,timestep,pattern_speed)
        else:
            return pattern_speed
        
    #rotation axis angle function + recenfunc are essentially direct copies
    #from Jhunt2021's code
    def rotation_axis_angle(self, axis):
        #function rotation_axis_angle, axis
        unitz=[0.0,0.0,1.0]
        angle=np.pi +(-1.0)*np.arccos(axis[2]/np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2))
        u = np.zeros(3)
        u[0]=-unitz[2]*axis[1]
        u[1]=unitz[2]*axis[0]
        u[2]=0.0
        n=np.sqrt(u[0]**2+u[1]**2)
        u=u/n
        kronecker=np.zeros(shape=(3,3))
        kronecker[0,0]=1.0
        kronecker[0,1]=0.0
        kronecker[0,2]=0.0
        kronecker[1,0]=0.0
        kronecker[1,1]=1.0
        kronecker[1,2]=0.0
        kronecker[2,0]=0.0
        kronecker[2,1]=0.0
        kronecker[2,2]=1.0
        ucrossu=np.zeros(shape=(3,3))
        for i in range (0,3) :
            for j in range (0,3):
                ucrossu[i,j]=u[i]*u[j]
        u_x=np.zeros(shape=(3,3))
        u_x[0,0]=0.0
        u_x[0,1]=(-1.0)*u[2]
        u_x[0,2]=u[1]
        u_x[1,0]=u[2]
        u_x[1,1]=0.0
        u_x[1,2]=(-1.0)*u[0]
        u_x[2,0]=(-1.0)*u[1]
        u_x[2,1]=u[0]
        u_x[2,2]=0.0
        rotation=np.zeros(shape=(3,3))
        rotation = ucrossu + np.cos(angle)*(kronecker-ucrossu)+ np.sin(angle)*u_x
        return rotation
    
    def recen_wholesnap(self, snaparr):
        #using masscutoff to select bulge and disk particles. Note that there
        #is an upper bound on the mass for the bulge flag, this ensures that dwarf 
        #star particles are excluded 
        bulge_flag = (snaparr['mass']>self.masscutoff)&(snaparr['mass']<700)
        disk_flag = (snaparr['mass'] < self.masscutoff) #disk particles are 160 or 185 msun
        x = snaparr['x'] - np.nanmedian(snaparr[bulge_flag]['x'])
        y = snaparr['y'] - np.nanmedian(snaparr[bulge_flag]['y'])
        z = snaparr['z'] - np.nanmedian(snaparr[bulge_flag]['z'])
        vx = snaparr['vx'] - np.nanmedian(snaparr[bulge_flag]['vx'])
        vy = snaparr['vy'] - np.nanmedian(snaparr[bulge_flag]['vy'])
        vz = snaparr['vz'] - np.nanmedian(snaparr[bulge_flag]['vz'])
        
        rr = np.sqrt(x[disk_flag]**2 + y[disk_flag]**2 + z[disk_flag]**2)
        
        Jdisk = np.zeros((3, len((x[disk_flag]))))
        Jdisk[0,:] = (y[disk_flag])*(vz[disk_flag])-(z[disk_flag])*(vy[disk_flag])
        Jdisk[1,:] = (z[disk_flag])*(vx[disk_flag])-(x[disk_flag])*(vz[disk_flag])
        Jdisk[2,:] = (x[disk_flag])*(vy[disk_flag])-(y[disk_flag])*(vx[disk_flag])
        rcyl=5.  # cylinder within which you want to align the orbital momentum axis
        wr4kpc= np.where((rr < rcyl))
        jtot=np.zeros(3)
        jtot[0]=np.sum(Jdisk[0,wr4kpc])
        jtot[1]=np.sum(Jdisk[1,wr4kpc])
        jtot[2]=np.sum(Jdisk[2,wr4kpc])
        magjtot=np.sqrt(jtot[0]**2+jtot[1]**2+jtot[2]**2)
        jtot_normalised=jtot/magjtot
        rot_align_J = self.rotation_axis_angle(jtot_normalised)
        
        #all stars! not just disk! Defined angular momentum above with
        #inner region of the disk, but apply correction to ALL stars
        post=np.vstack((x,y,z))
        velt=np.vstack((vx,vy,vz))

        post=np.matmul(rot_align_J,post).T
        velt=np.matmul(rot_align_J,velt).T
        post[:,2]*=-1.
        post[:,0]*=-1.
        velt[:,2]*=-1.
        velt[:,0]*=-1.

        fxx=np.argmax(x)
        oxx=x[fxx]
        oyy=y[fxx]


        xxx=post[:,0][fxx]
        yyy=post[:,1][fxx]

        dummy,ang1,dummy2=coords.rect_to_cyl(oxx,oyy,0.)
        dummy,ang2,dummy2=coords.rect_to_cyl(xxx,yyy,0.)

        ang=ang1-ang2

        snaparr['x'] = post[:,0]*np.cos(ang)-post[:,1]*np.sin(ang)
        snaparr['y'] = post[:,0]*np.sin(ang)+post[:,1]*np.cos(ang)
        snaparr['z'] = post[:,2]
        snaparr['vx'] = velt[:,0]*np.cos(ang)-velt[:,1]*np.sin(ang)
        snaparr['vy'] = velt[:,0]*np.sin(ang)+velt[:,1]*np.cos(ang)
        snaparr['vz'] = velt[:,2]
        return snaparr
    
    def computerecenparams(self, start=None,finish=None):
        '''
        Create a csv file with the info needed to re-center at each time step
        this is mostly so you can re-center particles that you're tracing over 
        time. you can load an individual snap in re-centered, or re-center after
        by using this info
        '''
        
        #check for times
        self.gettimes()

        #get start and finish
        if start is None:
            start = self.start
        if finish is None:
            finish = self.finish

        if finish is None:
            finish = len(self.times)
            
        outarr = np.empty((finish-start,1),dtype=[('t','d'),('tind','H'),
                                                  ('rot_align_J00','f'),
                                                  ('rot_align_J01','f'),
                                                  ('rot_align_J02','f'),
                                                  ('rot_align_J10','f'),
                                                  ('rot_align_J11','f'),
                                                  ('rot_align_J12','f'),
                                                  ('rot_align_J20','f'),
                                                  ('rot_align_J21','f'),
                                                  ('rot_align_J22','f'),
                                                  ('ang','f'),
                                                  ('x_med','f'),('y_med','f'), ('z_med','f'),
                                                 ('vx_med','f'), ('vy_med','f'), ('vz_med','f')])

        #load a time step
        for i, tt in enumerate(range(start, finish)):
            print(tt)
            snaparr = self.loadwholesnap(tt)
            print(snaparr['t'][0])
            #using masscutoff to select bulge and disk particles. Note that there
            #is an upper bound on the mass for the bulge flag, this ensures that dwarf 
            #star particles are excluded 
            bulge_flag = (snaparr['mass']>self.masscutoff)&(snaparr['mass']<700)
            disk_flag = (snaparr['mass'] < self.masscutoff) #disk particles are 160 or 185 msun
            
            x_med = np.nanmedian(snaparr[bulge_flag]['x'])
            y_med = np.nanmedian(snaparr[bulge_flag]['y'])
            z_med = np.nanmedian(snaparr[bulge_flag]['z'])
            vx_med = np.nanmedian(snaparr[bulge_flag]['vx'])
            vy_med = np.nanmedian(snaparr[bulge_flag]['vy'])
            vz_med = np.nanmedian(snaparr[bulge_flag]['vz'])
            
            #saving out the medians
            outarr[i]['t'] = snaparr['t'][0]
            outarr[i]['tind'] = int(tt)
            outarr[i]['x_med'] = x_med
            outarr[i]['y_med'] = y_med
            outarr[i]['z_med'] = z_med
            outarr[i]['vx_med'] = vx_med
            outarr[i]['vy_med'] = vy_med
            outarr[i]['vz_med'] = vz_med
            
            #getting angle -
            x = snaparr['x'] - x_med
            y = snaparr['y'] - y_med
            z = snaparr['z'] - z_med
            vx = snaparr['vx'] - vx_med
            vy = snaparr['vy'] - vy_med
            vz = snaparr['vz'] - vz_med
            
            #radius
            rr = np.sqrt(x[disk_flag]**2 + y[disk_flag]**2 + z[disk_flag]**2)
        
            Jdisk = np.zeros((3, len((x[disk_flag]))))
            Jdisk[0,:] = (y[disk_flag])*(vz[disk_flag])-(z[disk_flag])*(vy[disk_flag])
            Jdisk[1,:] = (z[disk_flag])*(vx[disk_flag])-(x[disk_flag])*(vz[disk_flag])
            Jdisk[2,:] = (x[disk_flag])*(vy[disk_flag])-(y[disk_flag])*(vx[disk_flag])
            rcyl=5.  # cylinder within which you want to align the orbital momentum axis
            wr4kpc= np.where((rr < rcyl))
            jtot=np.zeros(3)
            jtot[0]=np.sum(Jdisk[0,wr4kpc])
            jtot[1]=np.sum(Jdisk[1,wr4kpc])
            jtot[2]=np.sum(Jdisk[2,wr4kpc])
            magjtot=np.sqrt(jtot[0]**2+jtot[1]**2+jtot[2]**2)
            jtot_normalised=jtot/magjtot
            rot_align_J = self.rotation_axis_angle(jtot_normalised)
            #want to save this out
            outarr[i]['rot_align_J00'] = rot_align_J[0,0]
            outarr[i]['rot_align_J01'] = rot_align_J[0,1]
            outarr[i]['rot_align_J02'] = rot_align_J[0,2]
            outarr[i]['rot_align_J10'] = rot_align_J[1,0]
            outarr[i]['rot_align_J11'] = rot_align_J[1,1]
            outarr[i]['rot_align_J12'] = rot_align_J[1,2]
            outarr[i]['rot_align_J20'] = rot_align_J[2,0]
            outarr[i]['rot_align_J21'] = rot_align_J[2,1]
            outarr[i]['rot_align_J22'] = rot_align_J[2,2]
            #all stars! not just disk! Defined angular momentum above with
            #inner region of the disk, but apply correction to ALL stars
            post=np.vstack((x,y,z))
            velt=np.vstack((vx,vy,vz))

            post=np.matmul(rot_align_J,post).T
            velt=np.matmul(rot_align_J,velt).T
            post[:,2]*=-1.
            post[:,0]*=-1.
            velt[:,2]*=-1.
            velt[:,0]*=-1.

            fxx=np.argmax(x)
            oxx=x[fxx]
            oyy=y[fxx]


            xxx=post[:,0][fxx]
            yyy=post[:,1][fxx]

            dummy,ang1,dummy2=coords.rect_to_cyl(oxx,oyy,0.)
            dummy,ang2,dummy2=coords.rect_to_cyl(xxx,yyy,0.)
            
            #finally! last angle here!
            ang=ang1-ang2
            
            outarr[i]['ang'] = ang
            np.save(self.pointfpath+'centering_info',outarr)
        
        #save as a .npy in the pointer file path
        np.save(self.pointfpath+'centering_info',outarr)

        
        return outarr
    def apply_recen(self, arr):
        '''
        Takes recentering info array and
        recenters sourcearr values

        Inputs
        -----------------------

        arr is a sourcearr with commondtype of one particle over a time series
        '''
        self.recen_info = np.load(self.pointfpath+'centering_info.npy',allow_pickle=1,mmap_mode='r')

        recen_ts = self.recen_info['t']
        #find indexes where recentered array times = particle path times
        if len(recen_ts[:,0]) != len(arr['t']): #recen_ts[:,0]
            print('oh no - the number of time steps',
                  'in particle path array do not match recentering info array!',
                 'checking if path array is a subset of total time steps and proceeding')
            if len(recen_ts[:,0]) > len(arr['t']):
                #grabbing the indexes in the recentering info that have times that match those in the array
                recen_indx = np.array([np.where(recen_ts[:,0]==arr['t'][i])[0][0] for i in range(len(arr['t']))])
                tlen = len(arr['t'])
                cen_arr=np.empty(tlen,dtype=self.commondtype)

                for tt in range(len(arr['t'])):
                    #subtract out recetnered array medians from our particle paths
                    #at times where arrays match up
                    x = arr['x'][tt] - self.recen_info['x_med'][recen_indx][tt]
                    y = arr['y'][tt] - self.recen_info['y_med'][recen_indx][tt]
                    z = arr['z'][tt] - self.recen_info['z_med'][recen_indx][tt]
                    vx = arr['vx'][tt] - self.recen_info['vx_med'][recen_indx][tt]
                    vy = arr['vy'][tt] - self.recen_info['vy_med'][recen_indx][tt]
                    vz = arr['vz'][tt] - self.recen_info['vz_med'][recen_indx][tt]

                    #stack em up!
                    post=np.vstack((x,y,z))
                    velt=np.vstack((vx,vy,vz))

                    #reconstructing 3x3 J array
                    rot_align_J = np.zeros(shape=(3,3))
                    rot_align_J[0,0] = self.recen_info['rot_align_J00'][recen_indx][tt]
                    rot_align_J[0,1] = self.recen_info['rot_align_J01'][recen_indx][tt]
                    rot_align_J[0,2] = self.recen_info['rot_align_J02'][recen_indx][tt]
                    rot_align_J[1,0] = self.recen_info['rot_align_J10'][recen_indx][tt]
                    rot_align_J[1,1] = self.recen_info['rot_align_J11'][recen_indx][tt]
                    rot_align_J[1,2] = self.recen_info['rot_align_J12'][recen_indx][tt]
                    rot_align_J[2,0] = self.recen_info['rot_align_J20'][recen_indx][tt]
                    rot_align_J[2,1] = self.recen_info['rot_align_J21'][recen_indx][tt]
                    rot_align_J[2,2] = self.recen_info['rot_align_J22'][recen_indx][tt]


                    post=np.matmul(rot_align_J,post).T
                    velt=np.matmul(rot_align_J,velt).T
                    post[:,2]*=-1.
                    post[:,0]*=-1.
                    velt[:,2]*=-1.
                    velt[:,0]*=-1.

                    #apply the rotation + re-centering
                    ang = self.recen_info['ang'][recen_indx][tt]

                    cen_arr['x'][tt] = post[:,0]*np.cos(ang)-post[:,1]*np.sin(ang)
                    cen_arr['y'][tt] = post[:,0]*np.sin(ang)+post[:,1]*np.cos(ang)
                    cen_arr['z'][tt] = post[:,2]
                    cen_arr['vx'][tt] = velt[:,0]*np.cos(ang)-velt[:,1]*np.sin(ang)
                    cen_arr['vy'][tt] = velt[:,0]*np.sin(ang)+velt[:,1]*np.cos(ang)
                    cen_arr['vz'][tt] = velt[:,2]
                    cen_arr['mass'][tt] = arr['mass'][tt]
                    cen_arr['t'][tt] = arr['t'][tt]
                    cen_arr['idd'][tt] = arr['idd'][tt]
                if len(recen_ts[:,0]) < len(arr['t']):
                    print('input array has more times than recentering info - cannot proceed')

        else:
            #here we are assuming that both recen info and arr info are sorted by time, and that
            #the indexes align such that arr['t'][i] = recen_info['t'][i]
            tlen = len(arr['t'])
            cen_arr=np.empty(tlen,dtype=self.commondtype)

            for tt in range(len(arr['t'])):
                #subtract out recetnered array medians from our particle paths
                #at times where arrays match up
                x = arr['x'][tt] - self.recen_info['x_med'][tt]
                y = arr['y'][tt] - self.recen_info['y_med'][tt]
                z = arr['z'][tt] - self.recen_info['z_med'][tt]
                vx = arr['vx'][tt] - self.recen_info['vx_med'][tt]
                vy = arr['vy'][tt] - self.recen_info['vy_med'][tt]
                vz = arr['vz'][tt] - self.recen_info['vz_med'][tt]

                #stack em up!
                post=np.vstack((x,y,z))
                velt=np.vstack((vx,vy,vz))

                #reconstructing 3x3 J array
                rot_align_J = np.zeros(shape=(3,3))
                rot_align_J[0,0] = self.recen_info['rot_align_J00'][tt]
                rot_align_J[0,1] = self.recen_info['rot_align_J01'][tt]
                rot_align_J[0,2] = self.recen_info['rot_align_J02'][tt]
                rot_align_J[1,0] = self.recen_info['rot_align_J10'][tt]
                rot_align_J[1,1] = self.recen_info['rot_align_J11'][tt]
                rot_align_J[1,2] = self.recen_info['rot_align_J12'][tt]
                rot_align_J[2,0] = self.recen_info['rot_align_J20'][tt]
                rot_align_J[2,1] = self.recen_info['rot_align_J21'][tt]
                rot_align_J[2,2] = self.recen_info['rot_align_J22'][tt]


                post=np.matmul(rot_align_J,post).T
                velt=np.matmul(rot_align_J,velt).T
                post[:,2]*=-1.
                post[:,0]*=-1.
                velt[:,2]*=-1.
                velt[:,0]*=-1.

                #apply the rotation + re-centering
                ang = self.recen_info['ang'][tt]

                cen_arr['x'][tt] = post[:,0]*np.cos(ang)-post[:,1]*np.sin(ang)
                cen_arr['y'][tt] = post[:,0]*np.sin(ang)+post[:,1]*np.cos(ang)
                cen_arr['z'][tt] = post[:,2]
                cen_arr['vx'][tt] = velt[:,0]*np.cos(ang)-velt[:,1]*np.sin(ang)
                cen_arr['vy'][tt] = velt[:,0]*np.sin(ang)+velt[:,1]*np.cos(ang)
                cen_arr['vz'][tt] = velt[:,2]
                cen_arr['mass'][tt] = arr['mass'][tt]
                cen_arr['t'][tt] = arr['t'][tt]
                cen_arr['idd'][tt] = arr['idd'][tt]
        return cen_arr
    def apply_recen_snap(self, snaparr):
        '''
        Takes recentering info array and
        recenters snaparr values

        Inputs
        -----------------------

        centering snaparr for given time using the centering info
        '''
        self.recen_info = np.load(self.pointfpath+'centering_info.npy',allow_pickle=1,mmap_mode='r')
    
        recen_ts = self.recen_info['t']
        #find time step of snaparr in recen array

        tt = np.where(recen_ts[:,0] == snaparr['t'][0])[0][0]

        cen_arr=np.empty(len(snaparr),dtype=self.commondtype)

        #subtract out recetnered array medians from our particle paths
        #at times where arrays match up
        x = snaparr['x'] - self.recen_info['x_med'][tt]
        y = snaparr['y'] - self.recen_info['y_med'][tt]
        z = snaparr['z'] - self.recen_info['z_med'][tt]
        vx = snaparr['vx'] - self.recen_info['vx_med'][tt]
        vy = snaparr['vy'] - self.recen_info['vy_med'][tt]
        vz = snaparr['vz'] - self.recen_info['vz_med'][tt]

        #stack em up!
        post=np.vstack((x,y,z))
        velt=np.vstack((vx,vy,vz))

        #reconstructing 3x3 J array
        rot_align_J = np.zeros(shape=(3,3))
        rot_align_J[0,0] = self.recen_info['rot_align_J00'][tt]
        rot_align_J[0,1] = self.recen_info['rot_align_J01'][tt]
        rot_align_J[0,2] = self.recen_info['rot_align_J02'][tt]
        rot_align_J[1,0] = self.recen_info['rot_align_J10'][tt]
        rot_align_J[1,1] = self.recen_info['rot_align_J11'][tt]
        rot_align_J[1,2] = self.recen_info['rot_align_J12'][tt]
        rot_align_J[2,0] = self.recen_info['rot_align_J20'][tt]
        rot_align_J[2,1] = self.recen_info['rot_align_J21'][tt]
        rot_align_J[2,2] = self.recen_info['rot_align_J22'][tt]


        post=np.matmul(rot_align_J,post).T
        velt=np.matmul(rot_align_J,velt).T
        post[:,2]*=-1.
        post[:,0]*=-1.
        velt[:,2]*=-1.
        velt[:,0]*=-1.

        #apply the rotation + re-centering
        ang = self.recen_info['ang'][tt]

        cen_arr['x'] = post[:,0]*np.cos(ang)-post[:,1]*np.sin(ang)
        cen_arr['y'] = post[:,0]*np.sin(ang)+post[:,1]*np.cos(ang)
        cen_arr['z'] = post[:,2]
        cen_arr['vx'] = velt[:,0]*np.cos(ang)-velt[:,1]*np.sin(ang)
        cen_arr['vy'] = velt[:,0]*np.sin(ang)+velt[:,1]*np.cos(ang)
        cen_arr['vz'] = velt[:,2]
        cen_arr['mass'] = snaparr['mass']#[tt]
        cen_arr['t'] = snaparr['t']#[tt]
        cen_arr['idd'] = snaparr['idd']#[tt]
        return cen_arr
        
    def intobarframe(self,arr,frame=None,start=None,finish=None):
        '''
        Takes pattern_speeds array and selects the given [start:finish] and removes this rotation to move into bar frame for snaparr or sourcearr values. This now recenters the snap before moving into the bar frame!

        Inputs
        -----------------------
        start,finish [int] or frame: time indexes to select and adjust for -- could rewrite to read from arr instead of this...
        pattern_speeds [RAD/Gyr] (array or float): should automatically load within the procedure load
        
        arr is either:
            snaparr with commondtype of a single snap in time
        or:
            sourcearr with commondtype of one particle over a time series
            
        we've built in a flag to check if we are dealing with a snaparr or a sourcearr, and from there we
        
        apply the appropriate re-centering

        honestly this could be reorganized but idk if it would make it faster, just more readable
        '''
        self.patternspeeds = np.load(self.pointfpath+'patternspeed.npy',allow_pickle=1,mmap_mode='r')
        
        if arr['t'][0] == arr['t'][4]: #if the entries in the time column are the same, we have a snaparr
            arr = self.recen_wholesnap(arr)
        if arr['t'][0] != arr['t'][4]: #if entries in time column are different, we have a sourcearr  
            arr = self.apply_recen(arr) #recentering whole array
        #if working with a single whole snap
        print('returned array will be recentered and in the bar frame!')
        if frame is not None:
            start = frame
            finish = start+1
        
        #if loading a bunch use the simulation values 
        elif start is None:
            start = self.start
            if self.verbose>0:
                print('ATTN: start & finish nor frame set in intobarframe.')
            
            if len(np.shape(arr)) == 1:
                finish = start+1
            else:
                finish = self.finish

        if finish > len(self.patternspeeds):
            print('ATTN: Pattern speed array is shorter than requested time. Setting finish as ', str(len(self.patternspeeds)))
            finish = len(self.patternspeeds)
        #check that finish is at least high enough to check for the flips
        if start > 0:
            #if a timeseries, then check for the need to flip the angles after 2pi
            if start != finish-1:
                flipinds = np.where(abs(self.patternspeeds['barangle'][0:finish-1]-self.patternspeeds['barangle'][1:finish])>2)[0]+1
                sets = np.array(tuple(zip(flipinds[::2],flipinds[1::2])))
                pattern_speed_arr_grab = np.copy(self.patternspeeds)
                if len(sets)>0:
                    for s in sets:
                        pattern_speed_arr_grab['barangle'][s[0]:s[1]] += np.pi
                    lastflip = np.where(abs(self.patternspeeds['barangle'][sets[-1][1]:finish-1]-self.patternspeeds['barangle'][sets[-1][1]+1:finish])>2)[0]+1

                    if len(lastflip)>0:
                        pattern_speed_arr_grab['barangle'][(sets[-1][1]+lastflip)[0]:finish] += np.pi
                else:
                    print('ATTN: particle or time ',str(arr['idd']),str(start),'to',str(finish),' supposedly has no phi flips?')
                    
                pattern_speed_arr_tsampled = pattern_speed_arr_grab[start:finish]
            else:
                pattern_speed_arr_tsampled = np.copy(self.patternspeeds)[start:finish]
            #compute the rotation and the adjust all params, works on both snap and source array
            rotation = self.targetangle*u.deg.to(u.radian)-pattern_speed_arr_tsampled['barangle'].T#(pattern_speed_arr_tsampled['patternspeed_radperGyr'].T*arr['t'])[0].T #rad/Gyr*Gyr
            x_rot = arr['x'] * np.cos(rotation) - arr['y'] * np.sin(rotation)
            y_rot = arr['x'] * np.sin(rotation) + arr['y'] * np.cos(rotation)
            vx_rot = arr['vx'] * np.cos(rotation) - arr['vy'] * np.sin(rotation)
            vy_rot = arr['vx'] * np.sin(rotation) + arr['vy'] * np.cos(rotation)
            # vphi_rot = arr['vphi'] - (pattern_speed_arr_tsampled['patternspeed'].T*arr['r'])[0].T #km/s - km/s/kpc*kpc (pattern speed * radius of the star) for km/s 

            arr['x'] = x_rot
            arr['y'] = y_rot
            arr['vx'] = vx_rot
            arr['vy'] = vy_rot

            #_,arr['vphi'],_=coords.rect_to_cyl_vec(arr['vx'],arr['vy'],arr['vz'],arr['x'],arr['y'],arr['z'])
        #    _,arr['phi'],_=coords.rect_to_cyl(arr['x'],arr['y'],arr['z'])
        else:
            print('ATTN: Cannot preform bar frame adjustment on single the zeroth snap. Start: %i Finish: %i.'%(start,finish),flush=True)
        return arr

    
    def qcomputer(self,snaparr, timestepid, rmin, rmax, step, sim, verbose=0,savepath='',massflaglo=200,massflaghi=700):
        ''' 
        Use radial bins to characterize the individual time steps of the simulation.
        - Compute the rotation curve [approximated with vphi] from simulation output, 
        - Use the velocity curve to get epicyclic frequency, using the oort constant definition. 
        - Use these to compute the toomre Q parameter at each radial bin.

        Inputs 
        --------------------------------------------------

        Returns
        --------------------------------------------------

        '''
        if verbose >0:
            print('rotation curve at timestep', timestepid)
        if rmin - step <= 0:
            print('rmin - step <= 0, added small buffer to avoid NaNs')
            rmin = rmin + .0001
        

        #initalize the arrays
        radii = np.empty(int((rmax+step-rmin)/step))
        surface_densities = np.empty(int((rmax+step-rmin)/step))
        sigs = np.empty(int((rmax+step-rmin)/step))
        sigs_phi = np.empty(int((rmax+step-rmin)/step))
        velocity = np.empty(int((rmax+step-rmin)/step))
        velocity_r = np.empty(int((rmax+step-rmin)/step))
        mass_star = np.empty(int((rmax+step-rmin)/step))
        annuli_areas = np.empty(int((rmax+step-rmin)/step))

        #
        for ii,radius in enumerate(np.arange(rmin, rmax+step, step)):
            disk_annulus = (snaparr['r'] >= radius-step) & (snaparr['r'] < radius) & (snaparr['mass']<massflaglo)
            bulge_annulus = (snaparr['r'] >= radius-step) & (snaparr['r'] < radius) & (snaparr['mass']>massflaglo) & (snaparr['mass']<massflaghi)

            #mass
            mass_star[ii] = np.sum(snaparr['mass'][disk_annulus])

            #velocities
            velocity[ii] = np.nanmean(snaparr['vphi'][disk_annulus])
            velocity_r[ii] = np.nanmean(snaparr['vr'][disk_annulus])
            radii[ii] = radius

            #surface density here is gonna be mass per kpc^2
            sigs[ii] = np.nanstd(snaparr['vr'][disk_annulus])
            sigs_phi[ii] = np.nanstd(snaparr['vphi'][disk_annulus])

            # number of stars in annulus centered on given radius, with width of the step
            area = np.pi*((radius)**2 - (radius-step)**2)
            annuli_areas[ii] = area
            surface_densities[ii] = np.sum(snaparr['mass'][disk_annulus])/area #msun per kpc^2

        if verbose>3:
            print('mass_star shape', mass_star.shape, 'radii shape', radii.shape, 'velocity disp shape', \
                sigs.shape, 'surface density shape', surface_densities.shape, 'velocity shape', velocity.shape)
            print('circular velocities, vphi',velocity)
            print('radial velocities, vr',velocity_r)
            print('radial velocity dispersion', sigs)
            print('radial velocity dispersion', sigs_phi)
            print('surface density', surface_densities)
            print('disk stellar mass per annuli', mass_star)
            print('annuli areas', annuli_areas)
        #now we have rotation curve at each radius, next is to compute q using it
        dr = step #np.arange(rmin, rmax, step)
        dv_dr = np.gradient(velocity, dr)
        kappa = np.sqrt(2 * (velocity / (radii) + dv_dr) * (velocity / radii))
        #4.3 * 10**(-6) is G in msun, kpc, km/s, 3.36 is from definition of toomre q
        q = (kappa * sigs) / (3.36 * 4.3 * 10**(-6) * surface_densities)

        if verbose>3:
            print('dV/dR', dv_dr)
            print('q',q)

        v_circ_df = pd.DataFrame(np.array([velocity,velocity_r,radii, kappa, q, dv_dr, sigs, sigs_phi, surface_densities, mass_star, annuli_areas]).T, 
                            columns=['vel_circ','vel_r','radius','kappa','toomre_q','dvdr','vr_disp','vphi_disp','surface_densities','disk_mass','annuli_area'])
        v_circ_df.to_csv(savepath+'velocity_curve_toomreQ_timestep'+str(timestepid)+'.csv')

        return v_circ_df

    def getrotationcurve(self,timestepid,rmin=.25,rmax=30,step=.25,ncores=None, ntot=None, ndark=None, nstar=None, wdm=True,verbose=0,savepath='',simstr='',massflaglo=200,massflaghi=700): 
        #note want to set the high end of the mass flag here to be higher than the galaxy bulge and dm component, but lower than
        #the dwarf galaxy star and dm components. This way you can use the high-flag to remove the dwarf galaxy components
        #uses loadwholesnap() to load the stellar and dm arrays then computes rotation curve from the
        #individual mass components
        snaparr, snaparr_dark = self.loadwholesnap(timestepid,forcedarkmatter=True)
        
        self.gettimes()
        #initalize the arrays
        if verbose >0:
            print('rotation curve at timestep', timestepid)
        
        #getting r for snaparr and snaparr_dark
        #snaparr_r,snaparr_phi,snaparr_zz=coords.rect_to_cyl(snaparr['x'],snaparr['y'],snaparr['z'])
       #snaparr_dark_r,snaparr_dark_phi,snaparr_dark_zz=coords.rect_to_cyl(snaparr_dark['x'],snaparr_dark['y'],snaparr_dark['z'])

        #initalize the arrays
        radii = np.empty(int((rmax+step-rmin)/step))
        disk_enc = np.empty(int((rmax+step-rmin)/step))
        bulge_enc = np.empty(int((rmax+step-rmin)/step))
        halo_enc = np.empty(int((rmax+step-rmin)/step))
        total_enc = np.empty(int((rmax+step-rmin)/step))

        #
        for ii,radius in enumerate(np.arange(rmin, rmax+step, step)):
            disk_annulus = (snaparr['r'] < radius) & (snaparr['mass']<massflaglo)
            bulge_annulus = (snaparr['r'] < radius) & (snaparr['mass']>massflaglo) & (snaparr['mass']<massflaghi)
            dark_annulus = (snaparr_dark['r'] < radius) & (snaparr_dark['mass']<massflaghi)
            
            #disk_annulus = (snaparr_r < radius) & (snaparr['mass']<massflaglo)
            #bulge_annulus = (snaparr_r < radius) & (snaparr['mass']>massflaglo) & (snaparr['mass']<massflaghi)
            #dark_annulus = (snaparr_dark_r < radius) & (snaparr_dark['mass']<massflaghi)
            
            #mass enclosed
            disk_enc[ii] = np.sum(snaparr['mass'][disk_annulus])
            bulge_enc[ii] = np.sum(snaparr['mass'][bulge_annulus])
            dark_enc[ii] = np.sum(snaparr_dark['mass'][dark_annulus])
            total_enc[ii] = np.sum(snaparr['mass'][disk_annulus]) + np.sum(snaparr['mass'][bulge_annulus]) + np.sum(snaparr_dark['mass'][dark_annulus])
            
            #radius
            radii[ii] = radius
        if verbose>3:
            print('radius', radii)
            print('bulge mass enclosed',bulge_enc)
            print('dark mass enclosed',dark_enc)
            print('total mass enclosed',total_enc)
        #now we just do the math!
        G = 4.3 * 10**(-6) #kpc m_odot (km/s)^2
        disk_vc = np.sqrt((G * disk_enc)/radii)
        bulge_vc = np.sqrt((G * bulge_enc)/radii)
        dark_vc = np.sqrt((G * dark_enc)/radii)
        total_vc = np.sqrt((G * total_enc)/radii)

        v_circ_df = pd.DataFrame(np.array([radii, disk_vc, bulge_vc, dark_vc, total_vc]).T, 
                            columns=['radius','disk_vc', 'bulge_vc', 'dark_vc', 'total_vc'])
        v_circ_df.to_csv(savepath+'velocity_curve_timestep'+str(timestepid)+'.csv')

        return v_circ_df

   
    
    def profilesim(self):
        '''
        Function to fill in all the simulation paramters.
        '''
        self.zeroframe = self.loadwholesnap(0)

        while len(self.zeroframe[self.zeroframe['mass']>self.masscutoff]) == len(self.zeroframe):
            self.masscutoff *= 10

        self.loadpatternspeeds()

    def resetsim(self,newpath=None,sim = None,pointupdate=None,newpointerpath=None,keepzero=1):
        '''
        Function to reset and then fill params, will prompt for paths if not provided.
        '''
        self.nstar = None
        self.ndark = None
        self.ntot = None
        self.ncores = None
        self.times = None

        #new sim path
        if newpath is None:
            self.path = str(input('Provide new simulation path: '))
            if self.path[-1] != '/':
                self.path = self.path+'/'
        else:
            self.path = str(newpath)

        #new sim pointers
        if newpointerpath is not None:
            self.pointfpath = str(newpointerpath)
            if self.pointfpath[-1] != '/':
                self.pointfpath = self.pointfpath+'/'
        elif pointupdate is None:
            appendchoice = input('Would you like to (0) [leave pointer path], (1) [append] or (2) [provide new pointer path]?: ')
            if int(appendchoice) > 1:
                self.pointfpath = str(input('Provide full new pointer path: '))
            elif int(appendchoice) > 0:
                newdirect = str(input('Provide new end directory to append to pointer path: '))
                self.pointfpath = self.pointfpath+newdirect
                if self.pointfpath[-1] != '/':
                    self.pointfpath = self.pointfpath+'/'
        else:
            self.pointfpath = self.pointfpath+str(pointupdate)
            if self.pointfpath[-1] != '/':
                self.pointfpath = self.pointfpath+'/'
        if sim is not None:
            self.sim = sim
        #if there isn't pointfpath
        self.profilesim()

# %%
