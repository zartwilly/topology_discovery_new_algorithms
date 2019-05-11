#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:09:24 2019

@author: willy
"""

import os;
import time;
import math;
import random;
import numpy as np;
import pandas as pd
import itertools as it;

import genererMatA as geneMatA;
import fonctions_auxiliaires as fct_aux;
import algo_couverture as algoCouverture;

import defs_classes as def_class;
import creation_graphe as creat_gr;



from pathlib import Path;

INDEX_COL_MATE_LG = "sommets_aretes";
INDEX_COL_MAT_GR = "nodes";
NOM_MATE_LG = "matE_generer.csv";
NOM_MATE_LG_k_alpha = "matE_k_";
EXTENSION = ".csv"
NOM_MAT_GR = "mat_generer.csv";

###############################################################################
#               calculate Hamming distance -> DEBUT
###############################################################################
def calculate_hamming_distance(mat_LG, mat_LG_k):
    """
    identifier la liste des aretes differentes entre les graphes en parametres.
    
    aretes_ajout = aretes ajoutees a LG
    aretes_supp = aretes supprimees de LG
    """
    aretes_modifs = set(); aretes_ajout = set(); aretes_supp = set()
    if isinstance(mat_LG, pd.DataFrame) and isinstance(mat_LG_k, pd.DataFrame):
        aretes_LG = fct_aux.aretes(mat_LG, orientation=False, val_0_1=1);
        aretes_LG_k = fct_aux.aretes(mat_LG_k, orientation=False, val_0_1=1);
        
        aretes_ajout = aretes_LG.union(aretes_LG_k) - aretes_LG;
        aretes_supp = aretes_LG.union(aretes_LG_k) - aretes_LG_k;
        aretes_modifs = aretes_ajout.union(aretes_supp);
        
    elif isinstance(mat_LG, list) and isinstance(mat_LG_k, list):
        aretes_ajout = set(mat_LG).union(set(mat_LG_k)) - set(mat_LG);
        aretes_supp = set(mat_LG).union(set(mat_LG_k)) - set(mat_LG_k);
        
    elif isinstance(mat_LG, set) and isinstance(mat_LG_k, set):
        aretes_ajout = mat_LG.union(mat_LG_k) - mat_LG;
        aretes_supp = mat_LG.union(mat_LG_k) - mat_LG_k;
        
    
    aretes_modifs = aretes_ajout.union(aretes_supp);
    return aretes_modifs
    
###############################################################################
#               calculate Hamming distance -> FIN
###############################################################################


###############################################################################
#               add or/andremove k edges -> DEBUT
###############################################################################
def add_remove_edges(mat_LG, aretes_LG, k_erreur, prob):
    """
    ajouter/supprimer les aretes du graphe selon la valeur de prob
    si prob = 0 => suppression d'aretes uniquement.
    si prob = ]0,1[ => ajout et suppression d'aretes.
    si prob = 1 => ajout d'aretes uniquement.
    """
#    aretes_LG = None;
#    if isinstance(mat_LG, pd.DataFrame):
#        aretes_LG = fct_aux.aretes(mat_LG, orientation=False, val_0_1 = 1);
#    elif isinstance(mat_LG, list):
#        aretes_LG = mat_LG;
        
    aretes_modifiees = {"aretes_supprimees":[], 
                        "aretes_ajoutees":[]};
    mat_LG_k = mat_LG.copy()
    aretes_LG = list(aretes_LG)
    
    nbre_aretes_a_supp = math.ceil(k_erreur * (1 - prob));
    nbre_aretes_a_ajout = k_erreur - nbre_aretes_a_supp;
    for _ in range(0, nbre_aretes_a_supp):
        id_arete, arete = random.choice(list(enumerate(aretes_LG)))
        aretes_LG.pop(id_arete)
        mat_LG_k.loc[tuple(arete)[0], tuple(arete)[1]] = 0;
        mat_LG_k.loc[tuple(arete)[1], tuple(arete)[0]] = 0;
        aretes_modifiees["aretes_supprimees"].append(arete);
    
    not_aretes_LG = fct_aux.aretes(mat_LG, orientation=False, val_0_1 = 0);
    not_aretes_LG = list(not_aretes_LG)
    for _ in range(0, nbre_aretes_a_ajout):
        id_arete, arete = random.choice(list(enumerate(not_aretes_LG)))
        not_aretes_LG.pop(id_arete)
        aretes_LG.append(arete)
        mat_LG_k.loc[tuple(arete)[0], tuple(arete)[1]] = 1;
        mat_LG_k.loc[tuple(arete)[1], tuple(arete)[0]] = 1;
        aretes_modifiees["aretes_ajoutees"].append(arete);
        
    aretes_LG = set(aretes_LG)
    return mat_LG_k, aretes_LG, aretes_modifiees;  

###############################################################################
#               add or/andremove k edges -> FIN
###############################################################################


###############################################################################
#               execute_algos (couverture and correction)-> DEBUT
###############################################################################
def execute_algos(mat_GR, 
                   mat_LG, 
                   chemin_matrice, 
                   mode, 
                   critere, 
                   prob, 
                   k_erreur, 
                   nbre_graphe, 
                   num_graph_G_k,
                   alpha,
                   number_items_pi1_pi2,
                   DBG):
    """
    executer les algorithmes de couverture et de correction selon les parametres.
    """
    print("num_graph_G_k={} <=== debut ===>".format(num_graph_G_k))
    start_G_k = time.time()
    
    moy_dh, moy_dc = 0, 0;
    sum_dh, sum_dc = np.inf, np.inf;
    
    aretes_LG = fct_aux.aretes(mat_LG, orientation=False, val_0_1=1)
    
    for alpha_ in range(0,alpha):
        mat_LG_k_alpha, aretes_LG_k_alpha, aretes_modifiees = \
                                            add_remove_edges(
                                                    mat_LG, 
                                                    aretes_LG, 
                                                    k_erreur, 
                                                    prob)
        
        mat_LG_k_alpha.to_csv(chemin_matrice + \
                              NOM_MATE_LG_k_alpha + str(alpha_) + EXTENSION, 
                              index_label = INDEX_COL_MATE_LG)
        sommets_k_alpha = creat_gr.sommets_mat_LG(mat_LG_k_alpha);
        # algo couverture
#        cliques_couvertures, aretes_LG_k_alpha_res, sommets_k_alpha_res = \
#                                algoCouverture.clique_covers(
#                                    mat_LG_k_alpha, 
#                                    aretes_LG_k_alpha, 
#                                    sommets_k_alpha)
        # algo de correction
        
        # calcul distance
        dc_alpha = len(calculate_hamming_distance(
                                        mat_LG = aretes_LG, 
                                        mat_LG_k = aretes_LG_k_alpha))         # aretes_LG_k_alpha_ est celle apres correction. On peut ajouter aussi aretes_LG_k_alpha
        dh_alpha = len(calculate_hamming_distance(
                                        mat_LG = aretes_LG, 
                                        mat_LG_k = aretes_LG_k_alpha))
        
        sum_dc = dc_alpha if sum_dc == np.inf else sum_dc+dc_alpha;
        sum_dh = dh_alpha if sum_dh == np.inf else sum_dh+dh_alpha;
        pass #  for alpha_
    
    moy_dc = sum_dc/alpha;
    moy_dh = sum_dh/alpha;
    if moy_dh == 0 and moy_dc == 0:
        correl_dc_dh = 1;
    else:
        correl_dc_dh = abs(moy_dh - moy_dc) / max(moy_dh, moy_dc);
        
    # ecrire dans un fichier pouvant etre lu pendant qu'il continue d'etre ecrit
    nbre_sommets_LG = len(mat_LG.columns);
    chemin_dist = chemin_matrice+"../.."+"/"+"distribution"+"/"
    path = Path(chemin_dist); 
    path.mkdir(parents=True, exist_ok=True) if not path.is_dir() else None;
    
    f = open(chemin_dist + \
             "distribution_moyDistLine_moyHamming_k_" + \
             str(k_erreur) + \
             ".txt","a")
    f.write(str(num_graph_G_k) + ";" +\
            str(k_erreur) + ";" + \
            str( round(moy_dc,2) ) + ";" + \
            str( round(moy_dh,2) ) + ";" + \
#            str( len() ) + ";" + \
            str(nbre_sommets_LG) + ";" + \
            str(len(aretes_LG)) + ";" + \
            str(correl_dc_dh) + "\n"
            )
    
    print("num_graph_G_k={} <=== Termine :runtime={} ===>".format(
            num_graph_G_k, round(time.time()-start_G_k, 4)))
    pass  # execute_algo

###############################################################################
#               execute_algos (couverture and correction)-> FIN
###############################################################################


###############################################################################
#               define parameters for simulation -> DEBUT
###############################################################################
def define_parameters(dico_parametres):
    """
    definir les parametres pour chaque execution de la fonction execute_algos().
    """
    
    if dico_parametres["DBG"] :
            print("modes={}, criteres={}, probs={}, k_erreurs={}, nbre_graphes={}"\
                  .format(len(dico_parametres["modes_correction"]), 
                          len(dico_parametres["criteres_selection_compression"]), 
                          len(dico_parametres["probs"]), 
                          len(dico_parametres["k_erreurs"]), 
                          len(dico_parametres["nbre_graphes"]) ))
            
    graphes_GR_LG = list();

    for (mode, critere, prob, k_erreur, nbre_graphe) in it.product(
                    dico_parametres["modes_correction"],
                    dico_parametres["criteres_selection_compression"],
                    dico_parametres["probs"],
                    dico_parametres["k_erreurs"],
                    dico_parametres["nbre_graphes"]
                    ) : 
#        if dico_parametres["DBG"] :
#            print("mode={}, critere={}, prob={}, k_Erreur={}, nbre_graphe={}"\
#                  .format(mode, critere, prob, k_erreur, nbre_graphe))
            
        rep_base = dico_parametres["rep_data"] + "/" + \
                critere + "_sommets_GR_" + \
                            str(dico_parametres["nbre_sommets_GR"]) + "/" +\
                mode + "/" + \
                "data_p_" + str(prob) + "/" + \
                "G_" + str(nbre_graphe) + "_" + str(k_erreur) ;
        
        chemin_matrice = rep_base + "/" + "matrices" + "/";
        path = Path(chemin_matrice); path.mkdir(parents=True, exist_ok=True);
        
        mat_LG, mat_GR = creat_gr.generer_reseau(
                            nbre_sommets_GR = dico_parametres["nbre_sommets_GR"], 
                            nbre_moyen_liens = dico_parametres["nbre_moyen_liens"], 
                            chemin_matrice = chemin_matrice)
        
        num_graph = "G_" + str(nbre_graphe) + "_" + str(k_erreur)
        graphes_GR_LG.append( (mat_GR, 
                               mat_LG, 
                               chemin_matrice, 
                               mode, 
                               critere, 
                               prob, 
                               k_erreur, 
                               nbre_graphe, 
                               num_graph,
                               dico_parametres["alpha"],
                               dico_parametres["number_items_pi1_pi2"],
                               dico_parametres["DBG"])
                            )   
    return graphes_GR_LG;
    
###############################################################################
#               define parameters for simulation -> FIN
###############################################################################

###############################################################################
#               -> DEBUT
###############################################################################
###############################################################################
#               -> FIN
###############################################################################

if __name__ == '__main__':
    start = time.time();
    
    BOOL_TEST = True;
    DBG = True;
    log_file = "DEBUG.log";
    rep_data = "data"
    
    
    # valeurs initiales 
    NBRE_GRAPHE = 10;
    NBRE_SOMMETS_GR = 5
    NBRE_MOYEN_LIENS = (2,5)
    K_ERREUR_MIN = 0
    K_ERREUR_MAX = 5
    STEP_K_ERREUR = 1
    PROB_MIN = 0
    PROB_MAX = 1
    STEP_PROB = 0.5
    ALPHA = 1
    NUM_ITEM_Pi1_Pi2 = 0.5
    
    # caracteristiques des graphes GR
    nbre_sommets_GR = NBRE_SOMMETS_GR; #8#5;
    nbre_moyen_liens = NBRE_MOYEN_LIENS;
    nbre_graphes = range(1, NBRE_GRAPHE+1, 1)
    
    # nbre d'aretes a modifier
    k_erreur_min = K_ERREUR_MIN;
    k_erreur_max = K_ERREUR_MAX;
    step_k_erreur = STEP_K_ERREUR;
    k_erreurs = range(k_erreur_min, k_erreur_max+1, step_k_erreur);
    
    # aretes a modifier (prob = 0 => suppression, prob = 1 => ajout)
    # si suppression STEP_PROB > 1 ==> STEP_PROB = 1.1
    prob_min = PROB_MIN;
    prob_max = PROB_MAX;
    step_prob = STEP_PROB;
    probs = np.arange(prob_min, prob_max+0.1, step_prob);
    
    # mode de correction et critere de selection sommets a corriger
    modes_correction = ["aleatoire_sans_remise", 
                         "degre_min_sans_remise", 
                         "cout_min_sans_remise", 
                         "aleatoire_avec_remise", 
                         "degre_min_avec_remise", 
                         "cout_min_avec_remise"]
    criteres_selection_compression = ["voisins_corriges", 
                                       "nombre_aretes_corrigees", 
                                       "voisins_nombre_aretes_corrigees"];
    
    if BOOL_TEST :
        k_erreurs = range(k_erreur_min, 2, 1)
        modes_correction = ["aleatoire_sans_remise"]
        criteres_selection_compression = ["voisins_corriges"]
        step_prob = 1.1                                                         # je veux supprimer des aretes uniquement 
        probs = np.arange(prob_min, prob_max+0.1, step_prob)
    
    dico_parametres = {
            "rep_data": rep_data, "log_file":log_file, 
            "alpha":ALPHA, "nbre_graphes":nbre_graphes,
            "nbre_sommets_GR":nbre_sommets_GR, 
            "nbre_moyen_liens":nbre_moyen_liens,
            "k_erreurs":k_erreurs, 
            "probs":probs,
            "modes_correction":modes_correction,
            "criteres_selection_compression":criteres_selection_compression, 
            "number_items_pi1_pi2":NUM_ITEM_Pi1_Pi2, 
            "DBG": DBG
                       }
    
    graphes_GR_LG = list();
    graphes_GR_LG = define_parameters(dico_parametres);
    print("nbre_graphes : {} => OK?".format(len(graphes_GR_LG)))
    
    
    
    print("runtime : {}".format( time.time() - start))