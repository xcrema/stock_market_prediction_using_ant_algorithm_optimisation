from stocker import Stocker
import numpy as np
from numpy import inf
import matplotlib as plt
import pandas as pd
import sklearn
active=1
StockName = Stocker('AMZN')
no_of_days=365             #number of days to predict
if v==active:                #activate ACO
    iteration = 1000
    n_ants = 500000
    n_citys = 500000
    m = n_ants
    n = n_citys
    e = .5         #evaporation rate
    alpha = 2     #pheromone factor
    beta = 1      #visibility factor
    
    #calculating the visibility of the next city visibility(i,j)=1/d(i,j)
    
    visibility = 1/d
    visibility[visibility == inf ] = 0
    
    #intializing pheromne present at the paths to the cities
    
    pheromne = .1*np.ones((m.StockName,n.StockName.dates))
    
    #intializing the rute of the ants with size rute(n_ants,n_citys+1) 
    #note adding 1 because we want to come back to the source city
    
    rute = np.ones((m.StockName,n.StockName+1))
    
    for ite in range(iteration):
        
        rute[:,0] = 1          #initial starting and ending positon of every ants '1' i.e city '1'
        
        for i in range(m.StockName):
            
            temp_visibility = np.array(visibility)         #creating a copy of visibility
            
            for j in range(n.StockName-1):
                #print(rute)
                
                combine_feature = np.zeros(5)     #intializing combine_feature array to zero
                cum_prob = np.zeros(5)            #intializing cummulative probability array to zeros
                
                cur_loc = int(rute[i,j]-1)        #current city of the ant
                
                temp_visibility[:,cur_loc] = 0     #making visibility of the current city as zero
                
                p_feature = np.power(pheromne[cur_loc,:],beta)         #calculating pheromne feature 
                v_feature = np.power(temp_visibility[cur_loc,:],alpha)  #calculating visibility feature
                
                p_feature = p_feature[:,np.newaxis]                     #adding axis to make a size[5,1]
                v_feature = v_feature[:,np.newaxis]                     #adding axis to make a size[5,1]
                
                combine_feature = np.multiply(p_feature,v_feature)     #calculating the combine feature
                            
                total = np.sum(combine_feature)                        #sum of all the feature
                
                probs = combine_feature/total   #finding probability of element probs(i) = combine_feature(i)/total
                
                cum_prob = np.cumsum(probs)     #calculating cummulative sum
                #print(cum_prob)
                r = np.random.random_sample()   #random no in [0,1)
                #print(r)
                city = np.nonzero(cum_prob>r)[0][0]+1       #finding the next city having probability higher then random(r) 
                #print(city)
                
                rute[i,j+1] = city              #adding city to route 
               
            left = list(set([i for i in range(1,n.StockName+1)])-set(rute[i,:-2]))[0]     #finding the last untraversed city to route
            
            rute[i,-2] = left                   #adding untraversed city to route
           
        rute_opt = np.array(rute)               #intializing optimal route
        
        dist_cost = np.zeros((m,1))             #intializing total_distance_of_tour with zero 
        dist_min_loc = np.argmin(dist_cost)             #finding location of minimum of dist_cost
        dist_min_cost = dist_cost[dist_min_loc]         #finging min of dist_cost
        
        best_route = rute[dist_min_loc,:]               #intializing current traversed as best route
        pheromne = (1-e)*pheromne                    #evaporation of pheromne with (1-e)
        
        for i in range(m):
            
            s = 0
            for j in range(n-1):
                
                s = s + d[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1]   #calcualting total tour distance
            
            dist_cost[i]=s                      #storing distance of tour for 'i'th ant at location 'i' 
            for i in range(m):
                for j in range(n-1):
                    dt = 1/dist_cost[i]
                    pheromne[int(rute_opt[i,j])-1,int(rute_opt[i,j+1])-1] += dt  
                    if dt>pheromne:
                        max=stocker(rute.pheromne)
                    #updating the pheromne with delta_distance
                    #delta_distance will be more with min_dist i.e adding more weight to that route peromne
        def predict():
            m.StockName=stocker.max.pheromne;   #Upper Limit
            m2, m2.Stockname=stocker.max.pheromne
        def create_prophet_model(int):
            StockName=best_route.append;
            if pheromne > max:
                # Print the predicted price
                print('Predicted Price on {} = ${:.2f}'.format(
                    rute[len(future) - 1, 'ds'].date(), future.ix[len(future) - 1, 'yhat']))
    
                title = '%s Historical and Predicted Stock Price'  % self.symbol
            else:
                title = '%s Historical and Modeled Stock Price' % self.symbol
            print('Predicted Price on ',StockName.append_future(x),' = $',avg(sum(best_route)))
            plt.plot(StockName.dates, StockName.predict(dates), color='black', label='Observations')
            plt.plot(StockName.dates, StockName.predict(dates), color='green', label='Modeled')
            plt.plot(StockName.dates, StockName.predict(dates), color='green', label='Confidence Interval', attr='block')
                    
            plt.xlabel('Date')
            plt.ylabel('Price $')
            plt.title(StockName, 'Historical and Predicted Stock Price')
        
        def evaluate_prediction():
            #Accuracy measure
            sklearn.metrics.stocker.confusion_matrix(Stockname.Startdate('2017-03-27'), Stockname.Startdate('2018-03-27'))
            Startdate=random.StockName.date();
            Enddate=random.StockName.date(after=Startdate);
            plt.scatter(dates,prices, color='black', label='Data')
            plt.plot(StockName.dates, StockName.predict(dates), color='black', label='Observations')
            plt.plot(StockName.dates, StockName.predict(dates), color='black', label='Observations')
            plt.plot(StockName.dates, StockName.predict(dates), color='blue', label='Predicted', attr='bold')
            plt.plot(StockName.dates, StockName.predict(dates), color='yellow', label='Confidence Interval', attr='block')
            plt.plot(StockName.midpoint.random(dates), label='Prediction Start', color='red', attr='dotted')
                        
            plt.xlabel('Date')
            plt.ylabel('Price $')
            plt.title(StockName, 'Model Evaluation from ',Startdate, 'to ', Enddate)       
        
model, model_data = StockName.create_prophet_model(days=no_of_days)
StockName.evaluate_prediction()

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    