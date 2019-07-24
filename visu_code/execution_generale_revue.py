#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:50:19 2019

@author: willy
"""

import time;
import itertools as it;
import bokeh_revue as bkh;
import matplotlib_revue as matplib;

if __name__ == '__main__':
    start = time.time()
    reps = ["/home/willy/Documents/python_topology_learning_simulation/data_repeat_old"]
    reps = ["/home/willy/Documents/python_topology_learning_simulation/data_repeat"]
    
    nbre_sommets_graphes = [15]; #[15] or [12];
    probs = [0.5]
    modes_correction = ["lineaire_simul50Graphes_priorite_aucune"]
    criteres_correction = ["aleatoire"]
    
    #k_erreurs = [1, 2, 5, 10, 15, 20] 
    #k_erreurs = [1, 2, 5, 7]
    k_erreurs = [2,5,10,20]
    k_erreurs = [2,5,10,20,30,40]
    
    ### test correction_k_1
#    reps = ["/home/willy/Documents/python_topology_learning_simulation_debug/correction_k_1/data"]
#    probs = [0]
#    k_erreurs = [1]
    
    
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
    
    
    BOOL_MATPLOTLIB_BOKEH_PLOT = True#False#True # True : bokeh, False: matplotlib
    df_kerrs=None; 
    p = None;    
    for tuple_caracteristique in tuple_caracteristiques:
        if BOOL_MATPLOTLIB_BOKEH_PLOT:
            df_kerrs, p = bkh.distribution_bokeh_avec_runtime_baton(
                            *tuple_caracteristique);
#            df_kerrs, p = bkh.affichage_baton_runtime_meme_fichier_html(
#                            *tuple_caracteristique)
        else:
#            df_kerrs = matplib.distribution_matplotlib(*tuple_caracteristique)
            df_kerrs = matplib.distribution_matplotlib_with_chunksize_k(
                            *tuple_caracteristique)
#            
        
        pass
        
    print("runtime:{}".format(time.time() - start))