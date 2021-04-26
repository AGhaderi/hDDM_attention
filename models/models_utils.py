import pandas as pd
import hddm

#----------------------------------------------------------------------------
def modelp(data):  # spatial attention deos not free 
    model = hddm.HDDM(data, depends_on = {'v':['coherency', 'stimulus']}, bias = True, include={'st'})
    #find a good starting point which helps with the convergence.
    model.find_starting_values()
    # Create model and start MCMC sampling
    model.sample(100000, burn=10000, dbname='modelp_.db', db='pickle', thin =2)
   
    return model

def run_modelp(data):  
    params_fitted = modelp(data)
    res = pd.concat((pd.DataFrame([params_fitted.values]), 
                     pd.DataFrame([params_fitted.dic], columns={'dic'})), axis=1)
    #save completely modelp 
    params_fitted.save('modelp')
    #save csv file of modelp
    res.to_csv('modelp.csv')
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
def modelt(data):  # spatial attention depends on non-decision time 
    model = hddm.HDDM(data, depends_on = {'v':['coherency', 'stimulus'], 't':'spatial'}, bias = True, include={'st'})
    #find a good starting point which helps with the convergence.
    model.find_starting_values()
    # Create model and start MCMC sampling
    model.sample(100000, burn=10000, dbname='modelt_.db', db='pickle', thin =2)
   
    return model

def run_modelt(data):  
    params_fitted = modelt(data)
    res = pd.concat((pd.DataFrame([params_fitted.values]), 
                     pd.DataFrame([params_fitted.dic], columns={'dic'})), axis=1)
    #save completely modelt
    params_fitted.save('modelt')
    #save csv file of modelt
    res.to_csv('modelt.csv')
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
def modelz(data):  # spatial attention depends on starting point 
    model = hddm.HDDM(data, depends_on = {'v':['coherency', 'stimulus'], 'z':'spatial'}, bias = True, include={'st'})
    #find a good starting point which helps with the convergence.
    model.find_starting_values()
    # Create model and start MCMC sampling
    model.sample(100000, burn=10000, dbname='modelz_.db', db='pickle', thin =2)
   
    return model

def run_modelz(data):  
    params_fitted = modelz(data)
    res = pd.concat((pd.DataFrame([params_fitted.values]), 
                     pd.DataFrame([params_fitted.dic], columns={'dic'})), axis=1)
    #save completely modelz
    params_fitted.save('modelz')
    #save csv file of modelz
    res.to_csv('modelz.csv')
#----------------------------------------------------------------------------



#----------------------------------------------------------------------------
def modela(data):  # spatial attention depends on boundary decision
    model = hddm.HDDM(data, depends_on = {'v':['coherency', 'stimulus'], 'a':'spatial'}, bias = True, include={'st'})
    #find a good starting point which helps with the convergence.
    model.find_starting_values()
    # Create model and start MCMC sampling
    model.sample(100000, burn=10000, dbname='modela_.db', db='pickle', thin =2)
   
    return model

def run_modela(data):  
    params_fitted = modela(data)
    res = pd.concat((pd.DataFrame([params_fitted.values]), 
                     pd.DataFrame([params_fitted.dic], columns={'dic'})), axis=1)
    #save completely modela
    params_fitted.save('modela')
    #save csv file of modelt
    res.to_csv('modela.csv')
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
def modelv(data):  # spatial attention depends on drift-rate
    model = hddm.HDDM(data, depends_on = {'v':['coherency', 'stimulus', 'spatial']}, bias = True, include={'st'})
    #find a good starting point which helps with the convergence.
    model.find_starting_values()
    # Create model and start MCMC sampling
    model.sample(100000, burn=10000, dbname='modelv_.db', db='pickle', thin =2)
   
    return model

def run_modelv(data):  
    params_fitted = modelv(data)
    res = pd.concat((pd.DataFrame([params_fitted.values]), 
                     pd.DataFrame([params_fitted.dic], columns={'dic'})), axis=1)
    #save completely modelv
    params_fitted.save('modelv')
    #save csv file of modelt
    res.to_csv('modelv.csv')
#----------------------------------------------------------------------------

      
