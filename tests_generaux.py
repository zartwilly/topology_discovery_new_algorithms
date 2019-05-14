#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:24:10 2019

@author: willy
"""
import time;
import random;
import numpy as np;
import pandas as pd;
import creation_graphe as creat_gr;
import fonctions_auxiliaires as fct_aux;
import algo_couverture as algo_couv;
import graph_discovery_simulation as gr_disco_simi;


def test_sommets_mat_LG(nbre_sommets_GR = 5, 
                        nbre_moyen_liens = (2,5), 
                        chemin_matrice = "tests/matrices/") :
    mat_LG, mat_GR = creat_gr.generer_reseau(nbre_sommets_GR, 
                                    nbre_moyen_liens, 
                                    chemin_matrice)
    sommets = creat_gr.sommets_mat_LG(mat_LG);
    sommet = np.random.choice(mat_LG.columns)
    voisins = set(sommets[sommet].voisins);
    
    test = True
    while len(voisins) != 0:
        voisin = voisins.pop()
        if mat_LG.loc[sommet, voisin] != 1 :
            test = False;
            
    if test :
        print("TEST : sommets_mat_LG = OK")
    else :
        print("TEST : sommets_mat_LG = NOK")
    
def test_creer_graphe(nbre_sommets_GR = 5, 
                        nbre_moyen_liens = (2,5), 
                        chemin_matrice = "tests/matrices/",
                        nbre_graphe = 10):
    for num_graphe in range(nbre_graphe):
        chemin = chemin_matrice + str(num_graphe) + "/"
        mat_GR, mat_LG, sommets, aretes = creat_gr.creer_graphe(
                                            nbre_sommets_GR, 
                                            nbre_moyen_liens, 
                                            chemin)
        sommet = np.random.choice(mat_LG.columns)
        voisins = set(sommets[sommet].voisins);
        
        test = True
        while len(voisins) != 0:
            voisin = voisins.pop()
            if mat_LG.loc[sommet, voisin] != 1 :
                test = False;
                
        if test :
            print("TEST : creation graphe {} = OK".format(num_graphe))
        else :
            print("TEST : creation graphe {}= NOK".format(num_graphe))
    
def test_define_parameters(dico_parametres) :
    graphes_GR_LG = list();
    graphes_GR_LG = gr_disco_simi.define_parameters(dico_parametres);
    
    nbre_graphe_cal = len(dico_parametres["modes_correction"]) * \
                        len(dico_parametres["criteres_selection_compression"]) * \
                          len(dico_parametres["probs"]) * \
                          len(dico_parametres["k_erreurs"]) * \
                          len(dico_parametres["nbre_graphes"])
    if nbre_graphe_cal == len(graphes_GR_LG) :
        print("TEST : define_parameters {} = OK".format(nbre_graphe_cal))
    else:
        print("TEST : define_parameters {} = NOK".format(nbre_graphe_cal))
        
    return graphes_GR_LG;
    
def test_add_remove_edges(graphes_GR_LG):
    
    dico_df = dict();
    for graphe_GR_LG in graphes_GR_LG : 
        mat_LG = graphe_GR_LG[1];
        prob = graphe_GR_LG[5];
        k_erreur = graphe_GR_LG[6];
        num_graph = graphe_GR_LG[8]+"_p_"+str(prob);
        aretes_LG = fct_aux.aretes(mat_LG, orientation=False, val_0_1=1)
        mat_LG_k, aretes_LG_k, aretes_modifiees = gr_disco_simi.add_remove_edges(
                                                mat_LG, 
                                                aretes_LG, 
                                                k_erreur, 
                                                prob
                                                )
        
        ### test : a effacer
        aretes_LG_k_from_mat_LG_k = fct_aux.aretes(mat_LG_k, orientation=False, val_0_1=1)
        aretes_diff_from_mat_LG_k = aretes_LG_k_from_mat_LG_k - \
                        aretes_LG_k_from_mat_LG_k.intersection(aretes_LG_k)
        
        ### test avec boucle for ===> debut : a effacer
        aretes_ajouts_for , aretes_supps_for = set(), set();
        for arete in aretes_LG: 
            if arete not in aretes_LG_k:
                aretes_supps_for.add(arete)
        for arete_k in aretes_LG_k:
            if arete_k not in aretes_LG:
                aretes_ajouts_for.add(arete_k)
        ### test avec boucle for ===> fin
        
        aretes_ajout_LG_cal = aretes_LG.union(aretes_LG_k) - aretes_LG
        aretes_supp_LG_cal = aretes_LG_k.union(aretes_LG) - aretes_LG_k
        
                        
        res = ""
        if aretes_ajout_LG_cal == set(aretes_modifiees["aretes_ajoutees"]) and \
            aretes_supp_LG_cal == set(aretes_modifiees["aretes_supprimees"]) :
            res = "OK"
            print("TEST : add_remove_edges OK")
        else:
            res = "NOK"
            print("TEST : add_remove_edges NOK")
            
        dico_df[num_graph] = {
                "nbre_aretes_diff": len(aretes_LG) - len(aretes_LG_k),
                "nbre_aretes_diff_from_mat": len(aretes_diff_from_mat_LG_k),
                "prob":prob,            
                "k_erreur":k_erreur, 
                "res":res,
                "aretes_ajout_LG_cal": aretes_ajout_LG_cal,
                "aretes_supp_LG_cal": aretes_supp_LG_cal,
                "aretes_ajoutees": set(aretes_modifiees["aretes_ajoutees"]),
                "aretes_supprimees": set(aretes_modifiees["aretes_supprimees"]),
                "aretes_LG":aretes_LG,
                "aretes_LG_k":aretes_LG_k,
                "aretes_ajouts_for":aretes_ajouts_for, 
                "aretes_supps_for":aretes_supps_for
                              }
        
    df_test_ajout_supp_k_aretes = pd.DataFrame.from_dict(dico_df).T;
    return df_test_ajout_supp_k_aretes;

def test_calculate_hamming_distance(graphes_GR_LG):
    dico_df = dict();
    for graphe_GR_LG in graphes_GR_LG:
        mat_LG = graphe_GR_LG[1];
        prob = graphe_GR_LG[5];
        k_erreur = graphe_GR_LG[6];
        num_graph = graphe_GR_LG[8]+"_p_"+str(prob);
        aretes_LG = fct_aux.aretes(mat_LG, orientation=False, val_0_1=1)
        mat_LG_k, aretes_LG_k, aretes_modifiees = \
                                        gr_disco_simi.add_remove_edges(
                                                mat_LG, 
                                                aretes_LG, 
                                                k_erreur, 
                                                prob
                                            )
        
        aretes_modifs = gr_disco_simi.calculate_hamming_distance(mat_LG, 
                                                                 mat_LG_k)
        aretes_modifs_cal = set(aretes_modifiees["aretes_supprimees"]).\
                            union(set(aretes_modifiees["aretes_ajoutees"]))
        
        res = ""
        if aretes_modifs == aretes_modifs_cal :
            res = "OK"
            print("TEST : DH OK")
        else:
            res = "NOK";
            print("TEST : DH NOK")
        dico_df[num_graph] = {
                "prob":prob,            
                "k_erreur":k_erreur, 
                "res":res,
                "DH":len(aretes_modifs),
                "aretes_modifs":aretes_modifs,
                "aretes_modifs_cal":aretes_modifs_cal,
                "aretes_ajoutees":aretes_modifiees["aretes_ajoutees"],
                "aretes_supprimees":aretes_modifiees["aretes_supprimees"]
                }
        
    return pd.DataFrame.from_dict(dico_df).T;

def test_is_state_selected_node(graphes_GR_LG):
    """
    faire un test pour savoir l etat a  attribuer a ce graphe
    ensuite modifier a 0 et 3
    enfin tester si un sommet est a 0 ou 3
    """
    dico_df = dict(); 
    nbre_etats_3 = 3; nbre_etats_0 = 3;
    etats_possibles = [0,2,3]
    print("TEST is_state ET selected_node => debut")
    for graphe_GR_LG in graphes_GR_LG:
        
        mat_LG = graphe_GR_LG[1];
        prob = graphe_GR_LG[5];
        etat = random.choice(etats_possibles);
        sommets = creat_gr.sommets_mat_LG(mat_LG, etat=2);
        nom_sommets = list(sommets.keys())
        
        noms = []
        if etat == 0:
            for nbre_0 in range(nbre_etats_0):
                id_nom_som, nom_sommet = random.choice(list(
                                            enumerate(nom_sommets)))
                nom_sommets.pop(id_nom_som);
                noms.append(nom_sommet);
                sommet = sommets[nom_sommet]
                sommet.etat = 0;
        if etat == 3:
            for nbre_3 in range(nbre_etats_3):
                id_nom_som, nom_sommet = random.choice(list(
                                            enumerate(nom_sommets)))
                nom_sommets.pop(id_nom_som);
                noms.append(nom_sommet);
                sommet = sommets[nom_sommet];
                sommet.etat = 3;
        bool_state = algo_couv.is_state(sommets_k_alpha=sommets,
                                        critere0=0, critere3=3);
        nbre_sommet_0_2_3 = 0;
        for nom_sommet, sommet in sommets.items():
            if sommet.etat == etat:
                nbre_sommet_0_2_3 += 1;
        
        # sommet selectionne
        selected_sommet = algo_couv.selected_node(sommets_k_alpha= sommets,
                                                  critere0=0, critere3=3)
        
        is_som_etat_modif = False;
        if selected_sommet != None and selected_sommet.nom in noms:
            is_som_etat_modif = True;
        selected_nom_som = selected_sommet.nom if selected_sommet != None else None; 
        
        
        num_graph = graphe_GR_LG[8]+"_p_"+str(prob);
        dico_df[num_graph] = {
                "nbre_sommets":len(mat_LG.columns),
                "etat_choisi": etat,
                "nbre_sommet_0_2_3":nbre_sommet_0_2_3,
                "bool_state":bool_state,
                "noms_etat_modifs":noms,
                "selected_sommet":selected_nom_som,
                "is_som_etat_modif":is_som_etat_modif
                }
        pass # for
    print("TEST is_state ET selected_node => FIN")
    return pd.DataFrame.from_dict(dico_df).T;

def test_build_matrice_of_subgraph(graphes_GR_LG):
    dico = dict();
    print("TEST build_matrice_of_subgraph => debut")
    for graphe_GR_LG in graphes_GR_LG:
        mat_LG = graphe_GR_LG[1];
        prob = graphe_GR_LG[5];
        num_graph = graphe_GR_LG[8]+"_p_"+str(prob);
        sommets = creat_gr.sommets_mat_LG(mat_LG, etat=0);
        
        id_nom_som, nom_sommet_alea = random.choice(list(
                                            enumerate(sommets.keys())))
        mat_subgraph = algo_couv.build_matrice_of_subgraph(
                            sommet=nom_sommet_alea,
                            sommets_k_alpha=sommets)
        
        aretes_LG = fct_aux.aretes(mat_GR=mat_LG,
                                       orientation=False,
                                       val_0_1=1)
        aretes_subgraph = fct_aux.aretes(mat_GR=mat_subgraph,
                                       orientation=False,
                                       val_0_1=1)
        aretes_int = aretes_LG.intersection(aretes_subgraph);
        res=""
        if aretes_int == aretes_subgraph:
            res = "OK"
        else:
            res = "NOK"
        dico[num_graph]={"res":res}
    print("TEST build_matrice_of_subgraph => FIN")
    return pd.DataFrame.from_dict(dico).T;

def test_verify_cliques():
    sommet = "3_5"
    cliques = [{'1_5', '2_5', '3_5', '4_5'}, {'3_4', '3_5', '4_5'}];
    cliques_coh = [];
    bool_clique, bool_coherent, cliques_coh = algo_couv.verify_cliques(
                                                cliques = cliques,
                                                nom_sommet = sommet)
    print("bool_clique={} \n bool_coherent={} \n cliques_coh={}".format(
            bool_clique, bool_coherent, cliques_coh))
    
    cliques = [{'1_5', '2_5', '3_5', '4_5'}, {'3_4', '3_5', '4_5'}, 
               {'3_5','3_4','3_6'}];
    cliques_coh = [];
    bool_clique, bool_coherent, cliques_coh = algo_couv.verify_cliques(
                                                cliques = cliques,
                                                nom_sommet = sommet)
    print("bool_clique={} \n bool_coherent={} \n cliques_coh={}".format(
            bool_clique, bool_coherent, cliques_coh))

def test_execute_algos(graphes_GR_LG) :
    for graphe_GR_LG in graphes_GR_LG:
        gr_disco_simi.execute_algos(*graphe_GR_LG);
    
if __name__ == '__main__':
    start = time.time();
    
    nbre_sommets_GR = 6;
    nbre_moyen_liens = (2,5);
    chemin_matrice = "tests/matrices/";
    nbre_graphe = 10;
    
    test_sommets_mat_LG(nbre_sommets_GR, 
                        nbre_moyen_liens, 
                        chemin_matrice)

    test_creer_graphe(nbre_sommets_GR, 
                        nbre_moyen_liens, 
                        chemin_matrice)
    
    
    k_erreur_min, k_erreur_max = 0,5;
    k_erreurs = range(k_erreur_min, k_erreur_max, 1)
    modes_correction = ["aleatoire_sans_remise"]
    criteres_selection_compression = ["voisins_corriges"]
    prob_min, prob_max = 0,1; step_prob = 1.0;                                                         # je veux supprimer des aretes uniquement 
    probs = np.arange(prob_min, prob_max+0.1, step_prob)
    nbre_graphes = range(1, nbre_graphe+1, 1)
    ALPHA = 1; NUM_ITEM_Pi1_Pi2 = 0.5; DBG = True;
    rep_data = "tests"; log_file  = "", 
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
    graphes_GR_LG = test_define_parameters(dico_parametres);
    
    df_test_add_remove = test_add_remove_edges(graphes_GR_LG)
    
    df_DH = test_calculate_hamming_distance(graphes_GR_LG)
    
    df_is_state_select_node = test_is_state_selected_node(graphes_GR_LG);
    
    df_subgraph = test_build_matrice_of_subgraph(graphes_GR_LG);
    
    test_verify_cliques();
    #test_execute_algos(graphes_GR_LG)
    print("runtime : {}".format( time.time() - start))