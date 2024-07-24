#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:16:14 2023

@author: Nishanth Anandanadarajah <anandanadarajn2@nih.gov> allrights reserved

this script is created to function
intialisation for out_liers 


consider the right side half distribution as main distribution
as well as assume that is symmetric distribution when the obtained distribution is 
multinodal distribution


modified Created on Mon Apr  3 on to accomadate the vertical threhold line
in the GMM model


this unimodal multimodel is based on the sea born package need to be checked 
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751

from copy import deepcopy
from scipy.signal import argrelmax#, argrelmin
from sklearn.mixture import GaussianMixture



logger = logging.getLogger("uni_multi_funcs")
while logger.handlers:
     logger.handlers.pop()
c_handler = logging.StreamHandler()
# link handler to logger
logger.addHandler(c_handler)
# Set logging level to the logger
# logger.setLevel(logging.DEBUG) # <-- THIS!
logger.setLevel(logging.INFO)
logger.propagate = False

def plot_the_ditribution(data, x,y,ind_max,
                         num_bins=20,density=True,title='',
                         save_fig=False, save_fig_name=''):

    '''
        to plot the distribution of the histogram
            with the maximum annotated 
    '''
    x_max = x[ind_max]
    y_max = y[ind_max]
    
    # find first and second max values in y_max
    index_first_max = np.argmax(y_max)
    maximum_y = y_max[index_first_max]
    if len(y_max)>1:
        second_max_y = max(y for y in y_max if y!=maximum_y)
        index_second_max = np.where(y_max == second_max_y)
    
    # plot
    plt.hist(data, bins=num_bins, density=density, color='y')
    plt.scatter(x_max, y_max, color='b')
    plt.scatter(x_max[index_first_max], y_max[index_first_max], color='r')
    if len(y_max)>1:
        plt.scatter(x_max[index_second_max], y_max[index_second_max], color='g')
    plt.title(title)
    if save_fig:
        plt.savefig(save_fig_name+'.png', bbox_inches='tight', pad_inches=0.05)
    plt.show()



def plot_the_ditribution_with_GMM(data, x,y,ind_max,_y_recons,_y_recons_f,
                                  _y_gauss_1, _y_gauss_2, thrshold_val,
                                  factorize=True,
                                  num_bins=20,density=True,title='',
                                  save_fig=False, save_fig_name=''):

    x_max = x[ind_max]
    y_max = y[ind_max]
    
    # find first and second max values in y_max
    index_first_max = np.argmax(y_max)
    maximum_y = y_max[index_first_max]
    second_max_y = max(y for y in y_max if y!=maximum_y)
    index_second_max = np.where(y_max == second_max_y)
    if factorize:
        
        y_max_fac_recon =  maximum_y/_y_recons[np.argmax(_y_recons)]
        y_max_fac_recon_f =  maximum_y/_y_recons_f[np.argmax(_y_recons_f)]
        
        y_recons = np.dot(_y_recons,y_max_fac_recon)#*(len(data)/30)
        # y_recons_f = np.dot(_y_recons_f,y_max_fac_recon_f)#*(len(data)/30)
        
        

        y_recons_f1 = np.dot(_y_gauss_1,y_max_fac_recon_f)#*(len(data)/30)
        y_recons_f2 = np.dot(_y_gauss_2,y_max_fac_recon_f)#*(len(data)/30)


    else:
        y_recons=deepcopy(_y_recons)
        
        # y_recons_f =deepcopy(_y_recons_f)
        y_max_fac_recon=1
        y_max_fac_recon_f=1
        # print(y_max_fac_recon)

    _ = sns.histplot(data,kde=True)

    # plt.hist(data, bins=num_bins, density=True, color='y')
    plt.scatter(x_max, y_max, color='b')
    plt.scatter(x_max[index_first_max], y_max[index_first_max], color='g')
    plt.scatter(x_max[index_second_max], y_max[index_second_max], color='g')
    plt.plot(x, y_recons,color='r')
    # plt.plot(x, y_recons_f,'--',color='r')
    plt.plot(x, y_recons_f1,'--',color='r')
    plt.plot(x, y_recons_f2,'--',color='r')

    plt.axvline(x = thrshold_val, color = 'b', label = 'threhold-value')
    plt.title(title)
    if save_fig:
        plt.savefig(save_fig_name+'.png', bbox_inches='tight', pad_inches=0.05)
    plt.show()
    return y_max_fac_recon,y_max_fac_recon_f




def unimodal_multimodal_distribution_mode_checker(data, 
                                                 num_bins=20, density=True,
                                                 no_smoothing=False, persent_consider =20,
                                                 plot_on=False,title='',
                                                 save_fig=True, save_fig_name=''):
    '''
     return the number of modes presnet in the given set of data
 
    Inorder to do this first calculate the histogram
        num_bins=20 number of bins created while make the histogram

    no_smoothing: donot calculate the smoothing
    
        based on the persent_consider   half persent_consider %  >>>> bin_center  <<<<< half persent_consider %
        by default 10 %  >>>> bin_center  <<<<< 10 %
        
        to check the neighbour bins of the histogram 
        
    ****
        even though when this function encounter multiple modes in automatically consider smooting to calcualte the better
        
        
    if smooting is present then the smoothed o/p's nearest bins (order=1)
    (two: one from left and one from right)
    Is checked
    
    density = optional :this is for np.histogram
        If False, the result will contain the number of samples in each bin. 
        If True, the result is the value of the probability density function at the bin, 
        normalized such that the integral over the range is 1. Note that the sum of the histogram values will 
        not be equal to 1 unless bins of unity width are chosen; it is not a probability mass function.
    
    
    '''
    #since our analysis is based on seaborn distribution close the open figs
    '''
    if  the plt.close option is turned off the seaborn function based approach will make issue
    
    '''
    try:   
        plt.close('all')
    except:
        pass
    
    if no_smoothing:
        #percentage we consider
        den = 2*100/persent_consider
        order=np.max([int(num_bins/den),1])
        
        
        n_orgin, bins_origin = np.histogram(data, num_bins, density=density)        
        # find index of minimum between two modes
        #order is the number of points check near
        ind_max = argrelmax(n_orgin,order=order)
        
        # the histogram plot shows the results of P
        if plot_on:
            x = np.linspace(np.min(data), np.max(data), num= num_bins)
            plot_the_ditribution(data, x,n_orgin,ind_max, num_bins, density=density,
                                  title= title,save_fig=save_fig, save_fig_name=save_fig_name)
            # num_modes_pos
            return ind_max[0], bins_origin, n_orgin
        
    if ((not no_smoothing) or len(ind_max[0])>1):

        '''
        using the seaborn to create the smoothed histogram
        sns.histplot(data, kde=True)
        
        partial code credit
        # https://stackoverflow.com/questions/52517145/how-to-retrieve-all-data-from-seaborn-distribution-plot-with-mutliple-distributi
        '''
        myPlot = sns.histplot(data,kde=True)

        # Fine Line2D objects
        lines2D = [obj for obj in myPlot.findobj() if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"]
        # Retrieving x, y data
        x, y_mapped = lines2D[0].get_data()[0], lines2D[0].get_data()[1]
               
        # find index of minimum between two modes
        #order is the number of points check near
        order=1
        ind_max = argrelmax(y_mapped,order=order)
        if plot_on and len(ind_max[0])==1:
            plot_the_ditribution(data, x,y_mapped,ind_max, num_bins, density=density,
                                  title= title,save_fig=save_fig, save_fig_name=save_fig_name)

        #since the disttribution is going to plotted in the multimodal_z_map_corr_distribution_handler
        else:
            plt.close()

            
        return ind_max[0], x, y_mapped


def multimodal_z_map_corr_distribution_handler(data, ind_max, x, y_mapped, 
                                               factor_check=10,factor_check_consider_second=100,
                                               plot_on=False,title='',
                                               save_fig=True, save_fig_name='',tail_peak_warn_on=True):
    '''
    this function will take the multinominal distribution
    of z-mapped corr-coeff 
    
    when the len(ind_max)>1 only took the last distribution
    
    
    if the peak is too low-only consider as annomally raise as potential loose lead suspect
    like C4-O1 channel 21-0995_F_9.7_1_di
        C4- last part
        O1-channel seems fully red
    factor_check=10 this will check th esecond right peak maximum value
    
    '''
    tail_peak=False
    
    #first sort the indexes
    ind_max = np.sort(ind_max)
    
    # if we only take the correlation-z-mapped maximum values as the main peak 
    #choose the last peak value
    #since the indexes are sorted
    x_right_max_index =  ind_max[-1]
    '''
    if the peak is too low-only consider as annomally raise as potential loose lead suspect
    '''
    x_right_second_max_index =  ind_max[-2]
    if factor_check_consider_second > (y_mapped[x_right_second_max_index]/ y_mapped[x_right_max_index]) > factor_check:
        '''
        inorder to annotate the whole lead or raise the issue some value is assigned
        '''
        cutpoint = x[-4]
        # raise("need to build the loose lead suspect with full annotation")
        tail_peak= True
        logger.warning("tail peak issue " +title)

        if plot_on:
            myPlot = sns.histplot(data,kde=True)
            plt.title(title)
            if save_fig:
                plt.savefig(save_fig_name+'.png', bbox_inches='tight', pad_inches=0.05)
            plt.show()
    elif factor_check_consider_second <=   (y_mapped[x_right_second_max_index]/ y_mapped[x_right_max_index]):
        #means choose the whole distribution
        cutpoint = x[0]-1
        
        if plot_on:
            myPlot = sns.histplot(data,kde=True)
            plot_the_ditribution(data, x,y_mapped,ind_max, 
                                 title= title,save_fig=save_fig, save_fig_name=save_fig_name)
        if tail_peak_warn_on:
            logger.warning("tail peak skipped "+title)
        tail_peak=True
    else:
    
        right_x = x[x_right_max_index:]
        
        #this value will decide the cut-off point for the distribution
        cutpoint=x[x_right_max_index-len(right_x)]
        
        if plot_on:
            
            """
            choose the full right side distribution as main distribution
            """
            right_y_mapped = y_mapped[x_right_max_index:]
            
            # new_x = np.zeros((2*len(right_x)))
            new_y_dist = np.zeros((2*len(right_y_mapped)))
            
            #for x using the orginal x_values since this going to be used in the IQR
            new_x=x[x_right_max_index-len(right_x):]
            
            #just flipping and assigning the y mappings
            # new_y_dist[0:len(right_y_mapped)]=np.fliplr(right_y_mapped)
            new_y_dist[0:len(right_y_mapped)]=np.flip(right_y_mapped)
            new_y_dist[len(right_y_mapped):]=right_y_mapped
            
            
            _ = sns.histplot(data,kde=True)
            plt.plot(new_x,new_y_dist,'--',color='r')
            plt.title(title)
            if save_fig:
                plt.savefig(save_fig_name+'.png', bbox_inches='tight', pad_inches=0.05)
            plt.show()
            
    #selected distribution
    sel_corr_dist = []
    for p in data:
        if p>=cutpoint:
            sel_corr_dist.append(p)
    return sel_corr_dist






def tail_check_dist(ind_max, y_mapped,  factor_check=10, check_all_factors=True):
    
    '''
    if we encounter more than one peak
    First check the peak of the histogram with the right most peaks to ignore tail peaks
    
    
    check_all_factors: to check the peaks belongs to tail
    '''
    #first sort the indexes
    ind_max = np.sort(ind_max)
    y_max = y_mapped[ind_max]
    
    # find first and second max values in y_max
    index_first_max = np.argmax(y_max)
    maximum_y = y_max[index_first_max]
    # print("in one")

    if check_all_factors:
        # print("check_all_factors")
        calc_all_fact = y_max/maximum_y 
        inc_fac=[]
        for fac in range(0,len(calc_all_fact)):
            if calc_all_fact[fac]>(1/factor_check):
                inc_fac.append(fac)
                
        # print('calc_all_fact',calc_all_fact)
        # print('inc_fac ',inc_fac)
        ind_max =ind_max[inc_fac]
        
    else:
        # print("in two")
# 
        #recurrently copare the rightmost peak with the maximum peak to avoid the uninted tails
        #check thr right most point with the maximum counts distribution
        while len(ind_max)>1 and (maximum_y/y_max[-1])>factor_check:
            # print('ind_max before delete ',ind_max)
    
            ind_max = np.delete(ind_max,[-1])
            # print(' (maximum_y/y_max[-1])>factor_check:', (maximum_y/y_max[-1]))
            # print('ind_max: ',ind_max)
            y_max = y_mapped[ind_max]
            

    return  ind_max



def GMM_based_binomial_assum_thresh(data, ind_max, x, y_mapped,  threh_prob=0.5,
                                   density=True,
                                   plot_on=False,title='', save_fig=True, save_fig_name=''):
    '''
    Find the GMM
    with binomial distribution and find the threhold based on the probability
    
    
    and consider the right side distribution as error-less distibution 
    threh_prob: the thresh probbiliyu less than this threshold value will be annoated as outiler 
    only on the left side of the distribution (right side-before the mean)
    '''
    X=[[p] for p in data]
    
    '''
    paramters for GaussianMixture
    '''
    # The number of mixture components.
    n_components=2
    
    
    # covariance_type{‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
    # String describing the type of covariance parameters to use. Must be one of:
    
    # ‘full’: each component has its own general covariance matrix.
    # ‘tied’: all components share the same general covariance matrix.
    # ‘diag’: each component has its own diagonal covariance matrix.
    # ‘spherical’: each component has its own single variance.
    covariance_type='full'
    
    # The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
    tol=1e-3
    
    # Non-negative regularization added to the diagonal of covariance. Allows to assure that the covariance matrices are all positive.
    reg_covar=1e-6
    
    # The number of EM iterations to perform.
    max_iter=100
    
    # The number of initializations to perform. The best results are kept.
    n_init = 1
    
    # The method used to initialize the weights, the means and the precisions. String must be one of:
    # ‘kmeans’ : responsibilities are initialized using kmeans.
    # ‘k-means++’ : use the k-means++ method to initialize.
    # ‘random’ : responsibilities are initialized randomly.
    # ‘random_from_data’ : initial means are randomly selected data points.
    init_params='kmeans'
    
    
    
    GMM = GaussianMixture(n_components=n_components,
                          covariance_type=covariance_type,tol=tol,reg_covar=reg_covar,max_iter=max_iter,
                          n_init=n_init, init_params=init_params).fit(X) # Instantiate and fit the model
    # # to get the parameters
    # param = GMM.get_params(deep=True)
    
    if not GMM.converged_:# Check if the model has converged
        try:
            
            logger.warning("Since the algorithm not converged try with 2 times max iteration")
            GMM = GaussianMixture(n_components=n_components,
                          covariance_type=covariance_type,tol=tol,reg_covar=reg_covar,max_iter=2*max_iter,
                          n_init=n_init, init_params=init_params).fit(X) # Instantiate and fit the model
            if GMM.converged_:
                raise("GMM not converged")
        except:
            raise("GMM not converged")
    
    
    means = GMM.means_ 
    if plot_on:
        
        covariances = GMM.covariances_
        #this is only need to plots
        weights= GMM.weights_
        
        '''
        mapping by mixture model
        '''
        mu=means[0]
        #since the variance is same as covaraince due to one dimentional data
        var =covariances[0,0,0]
        y_gauss_1 =  get_guassian(x, mu, var)
        y_gauss_2 =  get_guassian(x, means[1], covariances[1,0,0])
        
        # # Store as dataframe 
        # data.append(pd.DataFrame({'x':x, 'y':y}))
        
        '''
        plot the histogram
        
        '''
        order=1
        n=y_mapped
        # find index of minimum between two modes
        #order is the number of points check near
        ind_max = argrelmax(n,order=order)
        #
        '''
        place the points based on the probability of presence with the mixture model weights
        '''
        X=[]
        y_recons=[]
        for i in range(0,len(x)):
            v=x[i]
            # X.append([p])
            obtain_prob = GMM.predict_proba([[v]])
            #here the weights means the mixture propostions
            obtain_prob_weights= weights*obtain_prob
            
            y_recons.append(np.dot(obtain_prob_weights, np.stack([y_gauss_1[i],y_gauss_2[i]]))[0])
            # y_recons.append(np.stack([y_gauss_1[i],y_gauss_2[i]]))
        # print('_y_gauss_1 defined')

        _y_guass_1 = y_gauss_1*weights[0]
        _y_gauss_2 = y_gauss_2*weights[1]
        # print('_y_gauss_1 defined')
        '''
         to check the plots visually whether the factorisation works
        '''    
            
        # _y_recons_i =np.dot(weights,np.stack([y_gauss_1,y_gauss_2]))*len(data)
        # _y_recons_f = np.dot(y_recons,len(data))
    
        # y_max_fac_recon,y_max_fac_recon_f=  plot_the_ditribution(x,y_mapped,ind_max,_y_recons_i,_y_recons_f,factorize=True)
        # y_max_fac_recon,y_max_fac_recon_f=  plot_the_ditribution(x,y_mapped,ind_max,_y_recons_i,_y_recons_f,factorize=False)
    
        _y_recons_i =np.dot(weights,np.stack([y_gauss_1,y_gauss_2]))
        _y_recons_f = np.dot(y_recons,1)

    '''
    
    to get the probability
    
    lets check the values from the rightside mean
    to left side when the probability fell less than some thresh
    '''
    
    right_dist_index=np.argmax(means)
    mean_right = means[right_dist_index][0]
    
    thrshold_val=0
    for x_val_index in range(len(x)-1,0,-1):
        if x[x_val_index]<mean_right:
            prob = GMM.predict_proba([[x[x_val_index]]])[0][right_dist_index]
            if prob<threh_prob:
                thrshold_val=x[x_val_index]
                break
    #since we pass through the IQR outlier detection to pass the maximum cut off
    max_value = np.max(x)+1#just add some vlaue to avoid
    if plot_on:
        y_max_fac_recon,y_max_fac_recon_f=  plot_the_ditribution_with_GMM(data, x,y_mapped,ind_max,_y_recons_i,_y_recons_f,
                                                                          _y_guass_1, _y_gauss_2,thrshold_val,
                                                                          factorize=True,
                                                                          density=density, title= title,save_fig=save_fig, save_fig_name=save_fig_name)
    return thrshold_val, max_value
            
def get_guassian(x, mu, var):
    '''
        This function will return the guassian distribution
    '''  

    sigma = np.sqrt(var)
    y_gauss =  (1/(sigma * np.sqrt(2 * np.pi)))*np.exp( - (x - mu)**2 / (2 * var)) 
    return y_gauss
