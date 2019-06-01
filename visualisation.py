#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:12:20 2019

@author: willy
"""
import math;
import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;
import itertools as it;
import fonctions_auxiliaires as fct_aux;
import matplotlib.font_manager as font_manager;

from pathlib import Path;
from scipy.stats import norm;
from matplotlib import lines, markers;

NBRE_ROWS = 3;
NBRE_COLS = 2;
NBRE_SOMMETS_1 = 2;

DISTRIB_ROOT_FILE = "distribution_moyDistLine_moyHamming_k_";
DISTRIB_EXT_FILE = ".txt";
RESUM_ROOT_FILE = "resumeExecution_";
REUM_EXT_FILE = ".csv";
DATA_P_REP = "data_p_";

NAMES_HEADERS_DISTRIB = ("G_k", "k_erreur","moy_dc","moy_dh","nbre_sommets_LG",\
                         "nbre_aretes_LG","correl_dc_dh")

###############################################################################
#               distribution avec seaborn => debut
###############################################################################
def create_dataframe(rep_dist, k_erreurs):
    """
    creer un dataframe contenant les caracteristiques des k_erreurs.
    """
    df = pd.DataFrame();
    f = lambda st: "_".join([st.split("_")[0], st.split("_")[1],
                             st.split("_")[3], st.split("_")[4]])
    for k_erreur in k_erreurs:
        names_headers = [name+"_"+str(k_erreur) 
                            for name in NAMES_HEADERS_DISTRIB if name != "G_k"];
        names_headers.insert(0,"G_k")
        df_k = pd.read_csv(
                    rep_dist+DISTRIB_ROOT_FILE+str(k_erreur)+DISTRIB_EXT_FILE,
                    names=names_headers,
                    sep=";"
                         );
        df_k["G_k"] = df_k['G_k'].apply(f);
        df_k["moy_dc_"+str(k_erreur)] = df_k["moy_dc_"+str(k_erreur)].astype(int);
        df_k["moy_dh_"+str(k_erreur)] = df_k["moy_dh_"+str(k_erreur)].astype(int);
        df_k["correl_dc_dh_"+str(k_erreur)] = df_k["correl_dc_dh_"+str(k_erreur)].astype(int);
        if df.empty:
            df = df_k;
        else:
            df = pd.merge(left=df,right=df_k,left_on='G_k',right_on="G_k")
    
    return df;
    pass

def distribution_seaborn(critere_correction, 
                         mode_correction,
                         prob, 
                         k_erreurs,
                         nbre_sommets_GR,
                         rep_):
    """
    visualiser les distributions de moy_dc, moy_dh, correl_dc_dh, cumul_dh.
    """
    rep = rep_ + "/" \
          + critere_correction +"_sommets_GR_" + str(nbre_sommets_GR) + "/" \
          + mode_correction + "/" \
          + DATA_P_REP + str(prob);
    rep_dist = rep + "/" + "distribution" + "/";
    
    df_kerrs = create_dataframe(rep_dist, k_erreurs);
    
    # configuration figure
    fig = plt.figure();
    fig, axarr = plt.subplots(len(k_erreurs), 4);
    fig.subplots_adjust(hspace=0.5, wspace=0.6)
    default_size = fig.get_size_inches()
    fig.set_size_inches( (default_size[0]*2.0, default_size[1]*3.5) )
    print("w =", default_size[0], " h = ",default_size[1])
    
    #plot 
    for ind, k_erreur in enumerate(k_erreurs):
        min_, max_ = 0, np.inf;
        max_ = max(df_kerrs["moy_dc_"+str(k_erreur)]) \
                if max(df_kerrs["moy_dc_"+str(k_erreur)]) >= max(df_kerrs["moy_dh_"+str(k_erreur)]) \
                else max(df_kerrs["moy_dh_"+str(k_erreur)])
        nb_inter = max_/2+1 if max_%2 == 0 else math.ceil(max_/2);
        bins = np.around((np.linspace(min_, max_, nb_inter)), decimals = 1);
        
        ## ind = 0, axarr = 0 --> moy_dc
        df_kerrs["moy_dc_"+str(k_erreur)].hist(bins = bins, ax = axarr[ind,0])
        mu = df_kerrs["moy_dc_"+str(k_erreur)].mean(); 
        sigma = df_kerrs["moy_dc_"+str(k_erreur)].std();
        axarr[ind,0].set(
                xlabel= "moy_distance_correction", \
                ylabel= "nombre_graphe", \
                title = "distance de correction pour \n" \
                    + str(k_erreur) \
                    +" arete(s) modifiee(s) \n $\mu=%.3f,\ \sigma=%.3f\ $ " \
                    %(mu, sigma)
                        );
        axarr[ind,0].axvline(x=k_erreur, color = 'r');
        axarr[ind,0].set_xticklabels(range(0,int(max_),2), rotation=90);
        axarr[ind,0].set_yticklabels(
                ['{:3.2f}%'.format(
                        x*100/df_kerrs["moy_dh_"+str(k_erreur)].count()) \
                    for x in axarr[ind,0].get_yticks()]
                                    );
        
        ## ind = 0, axarr = 1 --> moy_dh
        df_kerrs["moy_dh_"+str(k_erreur)].hist(bins = bins, ax = axarr[ind,1]);
        mu = df_kerrs["moy_dh_"+str(k_erreur)].mean(); 
        sigma = df_kerrs["moy_dh_"+str(k_erreur)].std();
        axarr[ind,1].set(
                xlabel= "moy_distance_hamming", \
                ylabel= "nombre_graphe", \
                title = "distance de Hamming pour \n"\
                    + str(k_erreur) \
                    + " arete(s) modifiee(s) \n $\mu=%.3f,\ \sigma=%.3f\ $." \
                    %(mu, sigma)
                        );
        axarr[ind,1].axvline(x=k_erreur, color = 'r')
        axarr[ind,1].set_yticklabels(
                ['{:3.2f}%'.format(
                        x*100/df_kerrs["moy_dh_"+str(k_erreur)].count()) \
                    for x in axarr[ind,1].get_yticks()]
                                    );
#        axarr[ind,1].set_xticklabels(bins, rotation=0)
        
        # ind=0, axarr = 2 --> correl_dl_dh
        data_sort = df_kerrs["correl_dc_dh_"+str(k_erreur)]\
                    .sort_values(ascending = True);
        axarr[ind,2].step(data_sort, data_sort.cumsum())
        axarr[ind,2].set(
                xlabel= "correlation_DC_DH", 
                ylabel= "cumulative correlation", \
                title = "fonction de repartition de \n"\
                        +"correlation entre moy_dl et moy_dh \n pour "\
                        +str(k_erreur)+" cases modifiees."
                        );
        axarr[ind,2].set_yticklabels(
                ['{:3.2f}%'.format(
                        x*100/df_kerrs["correl_dc_dh_"+str(k_erreur)].count()) \
                    for x in axarr[ind,2].get_yticks()]
                                    );
        
        # ind=0, axarr = 3 --> cumul_dh
        df_kerrs.sort_values(by = "moy_dh_"+str(k_erreur), 
                             ascending=True, 
                             axis = 0, 
                             inplace = True);
        df_kerrs["nb_graphe_dh<x"] = \
            df_kerrs["moy_dh_"+str(k_erreur)]\
            .apply( lambda x: \
                   df_kerrs["moy_dh_"\
                            +str(k_erreur)][df_kerrs["moy_dh_"\
                                            +str(k_erreur)] < x].count()/\
                   df_kerrs["moy_dh_"\
                            +str(k_erreur)].count())
#        print("--->k={}, cumul_dh => min = {}, max = {},".format(k_error, df["nb_graphe_dh<x"].min(), df["nb_graphe_dh<x"].max()))
        axarr[ind,3].set(
                xlabel= "moy_DH", 
                ylabel= "number graph moy_DH < x ", \
                title = "cumulative moy_dh pour \n"+str(k_erreur)+" cases modifiees");
        axarr[ind,3].step(df_kerrs["moy_dh_"+str(k_erreur)],
                          df_kerrs["nb_graphe_dh<x"]);
        axarr[ind,3].set_xticklabels(np.arange(0, 
                                               df_kerrs["moy_dh_"+str(k_erreur)].count(), 
                                               10), 
                                     rotation=45);   
        
    # save axarr
    fig.tight_layout();
    plt.grid(True);
    save_vizu = rep + "/../" + "visualisation";
    path = Path(save_vizu); path.mkdir(parents=True, exist_ok=True);
   
    fig.savefig(save_vizu+"/"+"distanceMoyenDLDH_k_"+"_".join(map(str,k_erreurs)) \
                +"_p_"+str(prob)+".jpeg", dpi=190);
    pass
###############################################################################
#               distribution avec seaborn => fin
###############################################################################

###############################################################################
#               distribution avec bokeh => debut
###############################################################################

###############################################################################
#               distribution avec bokeh => fin
###############################################################################

if __name__ == '__main__':
    reps = ["tests"];
    nbre_sommets_graphes = [6]
    probs = [0.0];
    k_erreurs = [1,2,3,4];
    modes_correction = ["aleatoire_sans_remise"];
    criteres_correction = ["nombre_aretes_corrigees"];
    
    tuple_caracteristiques = [];
    for crit_mod_prob_k_nbreGraph_rep in it.product(
                                        criteres_correction,
                                        modes_correction,
                                        probs,
                                        [k_erreurs],
                                        nbre_sommets_graphes,
                                        reps):
        critere = crit_mod_prob_k_nbreGraph_rep[0];
        mode = crit_mod_prob_k_nbreGraph_rep[1];
        prob = crit_mod_prob_k_nbreGraph_rep[2];
        k_erreurs = crit_mod_prob_k_nbreGraph_rep[3];
        nbre_sommets_graphe = crit_mod_prob_k_nbreGraph_rep[4];
        
        tuple_caracteristiques.append(crit_mod_prob_k_nbreGraph_rep)
        
    for tuple_caracteristique in tuple_caracteristiques:
        distribution_seaborn(*tuple_caracteristique);
        
        