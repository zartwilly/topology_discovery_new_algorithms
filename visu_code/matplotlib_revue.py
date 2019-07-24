#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 06:09:20 2019

@author: willy
"""

import math;
import time;
import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;
import itertools as it;
import fonctions_auxiliaires_visu as fct_aux_vis;
import matplotlib.font_manager as font_manager;

from pathlib import Path;
from scipy.stats import norm;
from matplotlib import lines, markers;

import seaborn as sns;
sns.set()

###############################################################################
#                       CONSTANTES  ===> debut
###############################################################################
NAMES_HEADERS = ["num_graph", "k_erreur", "alpha", "dc", "dh", 
                     "aretes_matE", "correl_dc_dh", "runtime"];
NAME_COURBES = ["DC","DH","CUMUL_CORREL","CUMUL_DH"];
DISTRIB_ROOT_FILE = "distribution_moyDistLine_moyHamming_k_";
DISTRIB_EXT_FILE = ".txt";
DATA_P_REP = "data_p_";
MUL_WIDTH = 2.5;
MUL_HEIGHT = 3.5;

MARKERS = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", 
               "P", "*", "h", "H", "+", "x", "X", "D", "d"]

BOOL_ANGLAIS = True;
###############################################################################
#                       CONSTANTES  ===> fin
###############################################################################

###############################################################################
#                       plot dc et dh  ===> debut
###############################################################################
def plot_dc_dh(moy_dc_dh, 
                df_grouped_k, 
                k_erreur, 
                ax):
    """
    representer l'histogramme des distances dc et dh
    """
    label_dc_dh = ""
    if moy_dc_dh == "dc":
        label_dc_dh = "moy_dc";
    elif moy_dc_dh == "dh":
        label_dc_dh = "moy_dh";
    
    mu = df_grouped_k[moy_dc_dh].mean(); 

    sigma = df_grouped_k[moy_dc_dh].std();
    title, xlabel, ylabel = fct_aux_vis.title_xlabel_ylabel_figure(
                                label_dc_dh, 
                                k_erreur, 
                                mu, 
                                sigma, 
                                BOOL_ANGLAIS);
    
    min_, max_ = 0, np.inf
    max_ = int(np.rint( max(df_grouped_k[moy_dc_dh]) ))
    min_ = int(np.rint( min(df_grouped_k[moy_dc_dh]) ))
    nb_inter = 1;
    bins = list(range(min_, max_+1, nb_inter))
    print("min_={}, max_={}, nb_inter={}".format(min_, max_, nb_inter))
    
    df_grouped_k[moy_dc_dh].hist(bins=bins, ax=ax);
    
    ax.set(xlabel= xlabel, \
           ylabel= ylabel, \
           title = title
           );
    print("bins={}, yticks={}, xticks={}, count={} \n ".format(
            bins,
            ax.get_yticks(), 
            ax.get_xticks(), 
            df_grouped_k[moy_dc_dh].count()) )
    ax.axvline(x=k_erreur, color = 'r', linestyle='--', linewidth= 2);
    ax.axvline(x=mu, color = 'yellow', linestyle='--', linewidth= 2);
    ax.set_xticks(ticks=bins, minor=True);
    ax.set_yticks(ticks=ax.get_yticks());
    ax.set_yticklabels(
                ['{:3.2f}%'.format(
                        x*100/df_grouped_k[moy_dc_dh].count()) \
                    for x in ax.get_yticks()]
                    );
    
    return ax;
    pass
###############################################################################
#                       plot dc et dh  ===> fin
###############################################################################

###############################################################################
#                       plot cumul correl et cumul dh  ===> debut
###############################################################################
def plot_cumul_correl_dc_dh(correl_dc_dh_cumul, 
                                 df_grouped_k, 
                                 k_erreur, 
                                 ax):
    """
    representer la fonction de repartition cumulee de 
        la correlation entre dc et dh -> correl_dc_dh
        la distance de hamming -> dh
    """
    label_cumul, new_col_label = "", ""
    if correl_dc_dh_cumul == "cumul_correl":
        label_cumul = "correl_dc_dh"
        new_col_label = "nb_graphe_correl"
    elif correl_dc_dh_cumul == "cumul_dh":
        label_cumul = "dh"
        new_col_label = "nb_graphe_dh"
        
    df_corr_dc_sorted = df_grouped_k.sort_values(
                        by=label_cumul, axis=0, ascending=True)
    
    mu = df_grouped_k['dh'].mean();
    
    df_corr_dc_sorted[new_col_label+"<x"] = \
            df_corr_dc_sorted[label_cumul].apply( lambda x: \
                             df_corr_dc_sorted[label_cumul][df_corr_dc_sorted[label_cumul]<x].count()/\
                             df_corr_dc_sorted[label_cumul].count()
                             )
            
    title, xlabel, ylabel = fct_aux_vis.title_xlabel_ylabel_figure(
                                correl_dc_dh_cumul, k_erreur, 0, 0, 
                                BOOL_ANGLAIS);
                                                       
    ax.plot(df_corr_dc_sorted[label_cumul],
            df_corr_dc_sorted[new_col_label+"<x"])
    ax.set(xlabel= xlabel,
           ylabel= ylabel,
           title = title
           );
    
    if correl_dc_dh_cumul == "cumul_dh":
        ax.axvline(x=k_erreur, color='r', linestyle='--', linewidth=2);
        ax.axvline(x=mu, color= 'yellow', linestyle='--', linewidth=2);
    elif correl_dc_dh_cumul == "cumul_correl":
        mu_corr = df_corr_dc_sorted[label_cumul].mean();
        ax.axvline(x=mu_corr, color= 'yellow', linestyle='--', linewidth=2);
        
    return ax;

###############################################################################
#                       plot cumul correl et cumul dh  ===> fin
###############################################################################

###############################################################################
#                       plot du tps d'execution des k_erreurs ===> debut
###############################################################################
def plot_tps_calcul(df_kerrs, k_erreurs, ax_run):
    """
    representer le temps d'execution pour chaque k_erreur
    """
    colors = ["red", "yellow", "blue", "green", "rosybrown", 
              "darkorange", "fuchsia", "grey"]
    
    df_grouped = df_kerrs.groupby("num_graph")\
                    ["dc",'dh','k_erreur','correl_dc_dh','runtime'].mean()
    title = "execution time by k added/deleted edges"
    xlabel = "graph number"
    ylabel = "runtime(s)"
    
    for ind_k_erreur, k_erreur in enumerate(k_erreurs):
        df_grouped_k = df_grouped[ df_grouped["k_erreur"] == k_erreur ]
        df_grouped_k = df_grouped_k.sort_values(by='runtime', axis=0)
        
        df_grouped_k['graph_number'] = np.arange(1, df_grouped_k.shape[0]+1);
        
        ax_run.plot(df_grouped_k['graph_number'],
            df_grouped_k["runtime"], 
            color=colors[ind_k_erreur], 
#            marker = MARKERS[ind_k_erreur],
            linewidth=2, 
            linestyle="--",
#            markeredgewidth= 0.01,
#            markersize=3.5
            )
        ax_run.legend(k_erreurs, loc='upper left', ncol = 2)
        
    ax_run.set(xlabel= xlabel,
                  ylabel= ylabel,
                  title = title
                  );
    return ax_run;
    pass
###############################################################################
#                       plot du tps d'execution des k_erreurs  ===> fin
###############################################################################

###############################################################################
#       plot du diagramme en baton des DH en fonction des k_erreurs => debut
###############################################################################
def plot_baton_dh(df_kerrs, k_erreurs, ax_bat_dh):
    """
    representation du diagramme en baton des DH en fonction des k_erreurs.
    """
    df_grouped = df_kerrs.groupby("num_graph")\
                    ["dc",'dh','k_erreur'].mean()
                    
    f = lambda row: "_".join([row["num_graph"].split("_")[0], 
                              row["num_graph"].split("_")[1]])
        
    df_grouped = df_grouped.reset_index()
    df_grouped['num_graph'] = df_grouped.apply(f, axis=1);
    df_grouped['k_erreur'] = df_grouped['k_erreur'].astype(int)
    df_grouped_sorted = df_grouped.sort_values(by=['k_erreur','dh'])
    
    df_grouped_new_index = df_grouped_sorted.set_index(['k_erreur','num_graph'])
    
    df_grouped_new_index['dh'].unstack().plot(
                        kind='bar', 
                        legend=False, 
                        color = 'yellow',
#                        colormap='Paired',
                        ax = ax_bat_dh)
    
    title = "Hamming distance by k added/deleted edges"
    xlabel = "k error"
    ylabel = "DH_k"
    ax_bat_dh.set(xlabel= xlabel,
                  ylabel= ylabel,
                  title = title
                  );
#    ax_bat_dh.patch.set_color('blue')
    return ax_bat_dh;

###############################################################################
#       plot du diagramme en baton des DH en fonction des k_erreurs => fin
###############################################################################

###############################################################################
#               distribution avec matplotlib avec 
#                   representation fonction de repartition cumule
#                   diagramme en baton des DH_k en fonction des k_erreurs 
#               ====> debut
###############################################################################
def distribution_matplotlib(critere_correction, 
                         mode_correction,
                         prob, 
                         k_erreurs,
                         nbre_sommets_GR,
                         rep_):
    """
    visualiser les distributions de moy_dc, moy_dh, correl_dc_dh, cumul_dh.
    """
    rep, rep_dist = "", "";
    df_kerrs = None;
    rep = rep_ + "/" \
          + mode_correction + "/" \
          + critere_correction + "_sommets_GR_" + str(nbre_sommets_GR) + "/" \
          + DATA_P_REP + str(prob);
    rep_dist = rep + "/" + "distribution" + "/";
    df_kerrs = fct_aux_vis.create_dataframe_data_revue(rep_dist, k_erreurs, 
                                               DISTRIB_ROOT_FILE,
                                               DISTRIB_EXT_FILE,
                                               NAMES_HEADERS);
    
                                                       
    # configuration figure
    fig = plt.figure();
    fig, axarr = plt.subplots(len(k_erreurs), len(NAME_COURBES));
    fig.subplots_adjust(hspace=0.02, wspace=0.02)
    default_size = fig.get_size_inches()
    fig.set_size_inches( (default_size[0]*MUL_WIDTH, 
                          default_size[1]*MUL_HEIGHT) )
    print("w =", default_size[0], " h = ",default_size[1])

                                                   
    for ind, k_erreur in enumerate(k_erreurs):
        print("k_erreur = {} => treating...".format(k_erreur))
        
        df_kerrs_k = df_kerrs[df_kerrs["k_erreur"] == k_erreur]
        df_grouped_numGraph_k = df_kerrs_k.groupby("num_graph")\
                                ["dc",'dh',"correl_dc_dh",'runtime'].mean()
        
        # moy_dc
        axarr[ind, 0] = plot_dc_dh("dc", 
                                 df_grouped_numGraph_k, 
                                 k_erreur, 
                                 axarr[ind, 0])
        
        # moy_dh
        axarr[ind, 1] = plot_dc_dh("dh", 
                                 df_grouped_numGraph_k, 
                                 k_erreur, 
                                 axarr[ind, 1])

        # cumul_dh
        axarr[ind, 2] = plot_cumul_correl_dc_dh("cumul_dh", 
                                 df_grouped_numGraph_k, 
                                 k_erreur, 
                                 axarr[ind, 2])
        
        # cumul_correl
        axarr[ind, 3] = plot_cumul_correl_dc_dh("cumul_correl", 
                                 df_grouped_numGraph_k, 
                                 k_erreur, 
                                 axarr[ind, 3])
                     
    # save axarr
    fig.tight_layout();
#    plt.grid(True);
    save_vizu = rep + "/../" + "visualisation";
    path = Path(save_vizu); path.mkdir(parents=True, exist_ok=True);
   
    fig.savefig(save_vizu+"/"+"distanceMoyenDLDH_k_"+"_".join(map(str,k_erreurs)) \
                +"_p_"+str(prob)+".jpeg", dpi=190);
                
    #plot runtime
    fig_run, ax_run = plt.subplots(1,1);
    fig_run.set_size_inches( (default_size[0]*1.5, 
                              default_size[1]*1.5) )
    ax_run = plot_tps_calcul(df_kerrs, k_erreurs, ax_run)
    
    #plot baton des dh
    fig_bat_dh, ax_bat_dh = plt.subplots(1,1);
    fig_bat_dh.set_size_inches( (default_size[0]*1.5, 
                              default_size[1]*1.5) )
    ax_bat_dh = plot_baton_dh(df_kerrs, k_erreurs, ax_bat_dh)
    
    #save ax_run, ax_bat_dh
    fig_run.savefig(save_vizu+"/"+"runtime_k_"+"_".join(map(str,k_erreurs)) \
                +"_p_"+str(prob)+".jpeg", dpi=190);
    fig_bat_dh.savefig(save_vizu+"/"+"baton_DH_k_"+"_".join(map(str,k_erreurs)) \
                +"_p_"+str(prob)+".jpeg", dpi=190);
                                      
    return df_kerrs;
###############################################################################
#               distribution avec matplotlib avec 
#                   representation fonction de repartition cumule
#                   diagramme en baton des DH_k en fonction des k_erreurs 
#               ====> fin
###############################################################################
    
###############################################################################
#               distribution avec matplotlib avec
#                   with chunksize k_erreurs
#                   representation fonction de repartition cumule
#                   diagramme en baton des DH_k en fonction des k_erreurs 
#               ====> debut
###############################################################################
def distribution_matplotlib_with_chunksize_k(critere_correction, 
                         mode_correction,
                         prob, 
                         k_erreurs,
                         nbre_sommets_GR,
                         rep_):
    """
    visualiser les distributions de moy_dc, moy_dh, correl_dc_dh, cumul_dh.
    """
    rep, rep_dist = "", "";
    df_kerrs = None;
    rep = rep_ + "/" \
          + mode_correction + "/" \
          + critere_correction + "_sommets_GR_" + str(nbre_sommets_GR) + "/" \
          + DATA_P_REP + str(prob);
    rep_dist = rep + "/" + "distribution" + "/";
    df_kerrs = fct_aux_vis.create_dataframe_data_revue(rep_dist, k_erreurs, 
                                               DISTRIB_ROOT_FILE,
                                               DISTRIB_EXT_FILE,
                                               NAMES_HEADERS);
    k_erreurs_chunks = [k_erreurs[:3], k_erreurs[3:]] 
                                                       
    # configuration figure
    for k_erreurs in k_erreurs_chunks:
        MUL_WIDTH, MUL_HEIGHT = 1.75, 2.5; #1.95, 2.5
        fig = plt.figure();
        fig, axarr = plt.subplots(len(k_erreurs), len(NAME_COURBES));
        fig.subplots_adjust(hspace=0.00, wspace=0.00)
        default_size = fig.get_size_inches()
        fig.set_size_inches( (default_size[0]*MUL_WIDTH, 
                              default_size[1]*MUL_HEIGHT) )
        print("w =", default_size[0], " h = ",default_size[1])
    
                                                       
        for ind, k_erreur in enumerate(k_erreurs):
            print("k_erreur = {} => treating...".format(k_erreur))
            
            df_kerrs_k = df_kerrs[df_kerrs["k_erreur"] == k_erreur]
            df_grouped_numGraph_k = df_kerrs_k.groupby("num_graph")\
                                    ["dc",'dh',"correl_dc_dh",'runtime'].mean()
            
            # moy_dc
            axarr[ind, 0] = plot_dc_dh("dc", 
                                     df_grouped_numGraph_k, 
                                     k_erreur, 
                                     axarr[ind, 0])
            
            # moy_dh
            axarr[ind, 1] = plot_dc_dh("dh", 
                                     df_grouped_numGraph_k, 
                                     k_erreur, 
                                     axarr[ind, 1])
    
            # cumul_dh
            axarr[ind, 2] = plot_cumul_correl_dc_dh("cumul_dh", 
                                     df_grouped_numGraph_k, 
                                     k_erreur, 
                                     axarr[ind, 2])
            
            # cumul_correl
            axarr[ind, 3] = plot_cumul_correl_dc_dh("cumul_correl", 
                                     df_grouped_numGraph_k, 
                                     k_erreur, 
                                     axarr[ind, 3])
                         
        # save axarr
        fig.tight_layout();
    #    plt.grid(True);
        save_vizu = rep + "/../" + "visualisation";
        path = Path(save_vizu); path.mkdir(parents=True, exist_ok=True);
       
        fig.savefig(save_vizu+"/"+"distanceMoyenDLDH_k_"\
                    +"_".join(map(str,k_erreurs)) \
                    +"_p_"+str(prob)+".jpeg", dpi=190);
                
    #plot runtime
    fig_run, ax_run = plt.subplots(1,1);
    fig_run.set_size_inches( (default_size[0]*1.5, 
                              default_size[1]*1.5) )
    ax_run = plot_tps_calcul(df_kerrs, k_erreurs, ax_run)
    
    #plot baton des dh
    fig_bat_dh, ax_bat_dh = plt.subplots(1,1);
    fig_bat_dh.set_size_inches( (default_size[0]*1.5, 
                              default_size[1]*1.5) )
    ax_bat_dh = plot_baton_dh(df_kerrs, k_erreurs, ax_bat_dh)
    
    #save ax_run, ax_bat_dh
    fig_run.savefig(save_vizu+"/"+"runtime_k_"+"_".join(map(str,k_erreurs)) \
                +"_p_"+str(prob)+".jpeg", dpi=190);
    fig_bat_dh.savefig(save_vizu+"/"+"baton_DH_k_"+"_".join(map(str,k_erreurs)) \
                +"_p_"+str(prob)+".jpeg", dpi=190);
                                      
    return df_kerrs;
###############################################################################
#               distribution avec matplotlib avec 
#                   with chunksize k_erreurs
#                   representation fonction de repartition cumule
#                   diagramme en baton des DH_k en fonction des k_erreurs 
#               ====> fin
###############################################################################

if __name__ == '__main__':
    start = time.time()
    
    probs = [0.5]
    criteres_correction = ["aleatoire"]
    modes_correction = ["lineaire_simul50Graphes_priorite_aucune"]
    nbre_sommets_graphes = [15];
    k_erreurs = [2,5,10,20] # [1, 2, 5, 10, 15, 20] 
    reps = ["/home/willy/Documents/python_topology_learning_simulation/data_repeat"]

    tuple_caracteristiques = [];
    for crit_mod_prob_k_nbreGraph_rep in it.product(
                                        criteres_correction,
                                        modes_correction,
                                        probs,
                                        [k_erreurs],
                                        nbre_sommets_graphes,
                                        reps
                                        ):
        critere = crit_mod_prob_k_nbreGraph_rep[0];
        mode = crit_mod_prob_k_nbreGraph_rep[1];
        prob = crit_mod_prob_k_nbreGraph_rep[2];
        k_erreurs = crit_mod_prob_k_nbreGraph_rep[3];
        nbre_sommets_graphe = crit_mod_prob_k_nbreGraph_rep[4];
        
        tuple_caracteristiques.append(crit_mod_prob_k_nbreGraph_rep)
    
    df_kerrs=None    
    for tuple_caracteristique in tuple_caracteristiques:
        df_kerrs = distribution_matplotlib(*tuple_caracteristique);
#        plot_runtime(*tuple_caracteristique);
#        plot_baton_dh(*tuple_caracteristique)
    print("runtime = {}".format(time.time() -start))