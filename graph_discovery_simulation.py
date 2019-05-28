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
import algo_correction as algoCorrection;

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
#           analyser les resultats d'une execution de Graphe ==> DEBUT
###############################################################################
def analyse_resultat(cliques_couvertures, 
                     sommets_k_alpha_res,
                     sommets_GR):
    """
    analyser les resultats d'une execution (cad algo de couverture ou correction)
    sur un graphe G_k_alpha.
    """
    f=lambda x: set(x.split("_"))
    
    sommets_trouves=[]
    for cliq in cliques_couvertures:
        aretes = list(map(f, cliq))
        sommet_commun = None;
        sommet_commun = set.intersection(*aretes);
        if sommet_commun != None and len(sommet_commun) == 1:
            sommets_trouves.append(sommet_commun.pop())
    
    # calculer le nombre de sommets ayant un etat specifique
    etat0, etat1, etat_1, etat2, etat3 = set(), set(), set(), set(), set();
    for nom_som, sommet in sommets_k_alpha_res.items():
        if sommet.etat == 0:
            etat0.add(nom_som);
        elif sommet.etat == 1:
            etat1.add(nom_som);
        elif sommet.etat == 2:
            etat2.add(nom_som);
        elif sommet.etat == 3:
            etat3.add(nom_som);
        elif sommet.etat == -1:
            etat_1.add(nom_som);
    
    sommets_absents = sommets_GR.union(sommets_trouves) - set(sommets_trouves)
    return sommets_trouves, sommets_absents, \
            etat0, etat1, etat_1, etat2, etat3
    pass
###############################################################################
#           analyser les resultats d'une execution de Graphe ==> FIN
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
    
    results_k_alpha = [];
    moy_dh, moy_dc = 0, 0;
    sum_dh, sum_dc = np.inf, np.inf;
    
    aretes_LG = fct_aux.aretes(mat_LG, orientation=False, val_0_1=1)
    
    for alpha_ in range(0,alpha):
        result_k_alpha = None;
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
        cliques_couvertures, aretes_LG_k_alpha_res, sommets_k_alpha_res = \
                                algoCouverture.clique_covers(
                                    mat_LG_k_alpha, 
                                    aretes_LG_k_alpha, 
                                    sommets_k_alpha, 
                                    DBG)
        sommets_trouves_couv=[]; sommets_absents_couv=set();
        etat0_couv, etat1_couv, etat_1_couv, etat2_couv, etat3_couv = \
                                        set(), set(), set(), set(), set();
        sommets_trouves_couv, sommets_absents_couv, \
        etat0_couv, etat1_couv, etat_1_couv, etat2_couv, etat3_couv = \
            analyse_resultat(cliques_couvertures,
                             sommets_k_alpha_res, 
                             set(mat_GR.columns))
            
        # algo de correction
        sommets_trouves_cor=[]; sommets_absents_cor=set();
        etat0_cor, etat1_cor, etat_1_cor, etat2_cor, etat3_cor = \
                                        set(), set(), set(), set(), set();
        aretes_LG_k_alpha_cor = []
        if fct_aux.is_exists_sommet(sommets=sommets_k_alpha, etat_1=-1):
            aretes_LG_k_alpha_cor = aretes_LG_k_alpha_res.copy();
            cliques_couvertures_1 = list(cliques_couvertures.copy());
            aretes_res_non_effacees = list(map(frozenset, 
                                               aretes_LG_k_alpha_res));
            cliques_couvertures_1.extend(aretes_res_non_effacees);
            sommets_tmp = creat_gr.sommets_mat_LG(mat_LG_k_alpha)
            sommets_k_alpha_1 = fct_aux.modify_state_sommets_mat_LG(
                                    sommets=sommets_tmp,
                                    sommets_res=sommets_k_alpha_res)
            cliques_couvertures_cor, \
            aretes_LG_k_alpha_cor, \
            sommets_k_alpha_cor, \
            cliques_par_nom_sommets_k_alpha_cor, \
            dico_sommets_corriges = \
                            algoCorrection.correction_algo(
                                cliques_couverture=set(cliques_couvertures_1),
                                aretes_LG_k_alpha=aretes_LG_k_alpha,
                                sommets_LG=sommets_k_alpha_1,
                                mode_correction=mode,
                                critere_correction=critere,
                                number_items_pi1_pi2=number_items_pi1_pi2,
                                DBG=DBG
                                      )
            sommets_trouves_cor, sommets_absents_cor, \
            etat0_cor, etat1_cor, etat_1_cor, etat2_cor, etat3_cor = \
            analyse_resultat(cliques_couvertures_cor,
                             sommets_k_alpha_cor, 
                             set(mat_GR.columns))
        
        # calcul distance
        dc_alpha = len(calculate_hamming_distance(
                                        mat_LG = aretes_LG_k_alpha, 
                                        mat_LG_k = aretes_LG_k_alpha_cor))         # aretes_LG_k_alpha_ est celle apres correction. On peut ajouter aussi aretes_LG_k_alpha
        dh_alpha = len(calculate_hamming_distance(
                                        mat_LG = aretes_LG, 
                                        mat_LG_k = aretes_LG_k_alpha_cor))
        
        #resultat d'un execution k_alpha
        result_k_alpha = (num_graph_G_k,k_erreur,alpha_,mode,critere,prob,
                len(sommets_trouves_couv),len(sommets_absents_couv),
                len(etat0_couv),len(etat1_couv),len(etat_1_couv),
                len(etat2_couv),len(etat3_couv),
                len(sommets_trouves_cor),len(sommets_absents_cor),
                len(etat0_cor),len(etat1_cor),len(etat_1_cor),
                len(etat2_cor),len(etat3_cor),
                          dc_alpha, dh_alpha
                          )
        results_k_alpha.append(result_k_alpha);
        
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
    
    print("results_k_alpha={}".format(len(results_k_alpha)))
    print("num_graph_G_k={} <=== Termine :runtime={} ===>".format(
            num_graph_G_k, round(time.time()-start_G_k, 4)))
    if DBG :
        return results_k_alpha;
    else:
        return [];
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
        
        num_graph = "G_" + str(nbre_graphe) + "_" + str(k_erreur) + "_p_" +\
                    "".join(str(prob).split('.'))
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