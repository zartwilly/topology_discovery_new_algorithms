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

from bokeh.plotting import *
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, gridplot;
from bokeh.core.properties import value
from bokeh.palettes import Spectral5

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
                            nom_sommet=nom_sommet_alea,
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

def test_update_edges_neighbor(graphes_GR_LG):
    dico_df = dict();
    for graphe_GR_LG in graphes_GR_LG:
        mat_LG = graphe_GR_LG[1];
        prob = graphe_GR_LG[5];
        num_graph = graphe_GR_LG[8]+"_p_"+str(prob);
        sommets = creat_gr.sommets_mat_LG(mat_LG, etat=0);
        
        id_nom_som, nom_sommet_alea = random.choice(list(
                                            enumerate(sommets.keys())))
        aretes_LG = fct_aux.aretes(mat_GR=mat_LG,
                                   orientation=False,
                                   val_0_1=1)
#        print("TEST update_edge num_graph={}".format(num_graph));
        cliques = algo_couv.partitionner(
                                sommet = sommets[nom_sommet_alea],
                                sommets_k_alpha = sommets,
                                aretes_LG_k_alpha = aretes_LG,
                                DBG= True
                                )
        cliques_coh = []
        bool_clique, bool_coherent, cliques_coh = \
                            algo_couv.verify_cliques(
                                        cliques = cliques,
                                        nom_sommet = nom_sommet_alea)                    
        C1, C2 = set(), set(); 
        if len(cliques_coh) == 1:
            C1 = cliques_coh[0];
        elif len(cliques_coh) == 2:
            C1, C2 = cliques_coh[0], cliques_coh[1];
        aretes_LG_res, sommets = algo_couv.update_edges_neighbor(
                                                    C1 = C1,
                                                    C2 = C2,
                                                    aretes = aretes_LG,
                                                    sommets = sommets)
        aretes_supps_res = aretes_LG.union(aretes_LG_res) - aretes_LG_res;
        
        # transform sommets to dataframe puis calculer aretes_restantes et 
        # comparer aretes_restantes avec aretes_LG
        aretes_supps_cal = set();
        mat_res = fct_aux.convert_sommet_to_df(sommets_k_alpha=sommets);
        aretes_restantes = fct_aux.aretes(mat_GR=mat_res,
                                   orientation=False,
                                   val_0_1=1)
        aretes_supps_cal = aretes_LG.union(aretes_restantes) - aretes_restantes;
        res = ""
        if aretes_supps_cal == aretes_supps_res:
            res = 'OK';
        else:
            res = 'NOK';
        
        dico_df[num_graph] = {"nom_sommet":nom_sommet_alea,
               "voisins":set(sommets[nom_sommet_alea].voisins),
               "cliques":cliques,
               "cliques_coh":cliques_coh,
               "aretes_supps_res": aretes_supps_res,
               "aretes_supps_cal": aretes_supps_cal,
               "res":res
               }
        print("TEST update_edge, num_graph={}, res={}".format(num_graph,res));
    return pd.DataFrame.from_dict(dico_df).T;
    
def test_algo_covers(graphes_GR_LG) : 
    """
    test la fonction algo_cover et aussi is_exists_sommet
    """
    f=lambda x: set(x.split("_"))
    etat_recherche_1 = -1#0,1,2,3,-1;
    
    dico_df = dict()
    for graphe_GR_LG in graphes_GR_LG:
        prob = graphe_GR_LG[5];
        num_graph = graphe_GR_LG[8]+"_p_"+str(prob);
        
        mat_LG = graphe_GR_LG[1];
        aretes_LG = fct_aux.aretes(mat_LG); 
        sommets_LG = creat_gr.sommets_mat_LG(mat_LG)
        start = time.time()
        #test cliques_covers
        cliqs_couverts, aretes, sommets = \
                       algo_couv.clique_covers(mat_LG, aretes_LG, 
                                               sommets_LG,True);
        # test is_exists_sommet
        exist_som_1 = None;
        exist_som_1 = fct_aux.is_exists_sommet(sommets=sommets, 
                                               etat_1=etat_recherche_1)
        # test modify_state_sommets_mat_LG => pas fait car je ne sais pas ce que je dois comparer
#        sommets_tmp = creat_gr.sommets_mat_LG(mat_LG);
#        sommets_LG_after = fct_aux.modify_state_sommets_mat_LG(
#                                sommets=sommets_tmp,
#                                sommets_res=sommets)
        runtime = round(time.time() - start, 2);
        
        som_trouves=[]
        for cliq in cliqs_couverts:
            aretes = list(map(f, cliq))
            sommet_commun = None;
            sommet_commun = set.intersection(*aretes);
            if sommet_commun != None and len(sommet_commun) == 1:
                som_trouves.append(sommet_commun.pop())
        
        # calculer le nombre de sommets ayant un etat specifique
        etat0, etat1, etat_1, etat2, etat3 = set(), set(), set(), set(), set();
        for nom_som, sommet in sommets.items():
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
        
        mat_GR = graphe_GR_LG[0]
        som_GR = set(mat_GR.columns)
        som_absents = som_GR.union(som_trouves) - set(som_trouves)
              
        res = ""
        if som_GR == set(som_trouves):
            res = 'OK'
        else:
            res = 'NOK'
        
        print("TEST cliques_cover num_graphe={} runtime={}, ==>res={}, exist_som={}".format(
                num_graph,runtime,res, exist_som_1))
        
        dico_df[num_graph] = {"res":res,
                           "nbre_som_GR":len(som_GR),
                           "nbre_som_trouves":len(som_trouves),
                           "som_absents":som_absents,
                           "aretes_res":aretes,
                           "etat0": len(etat0),
                           "etat1": len(etat1),
                           "etat2": len(etat2),
                           "etat3": len(etat3),
                           "etat_1": len(etat_1),
                           "exist_som_1": exist_som_1,
                           "runtime":runtime
               }
        
    return pd.DataFrame.from_dict(dico_df).T;


def test_execute_algos(graphes_GR_LG) :
    for graphe_GR_LG in graphes_GR_LG:
        gr_disco_simi.execute_algos(*graphe_GR_LG); 
    pass    
        
###############################################################################
#                          plot in bokeh --> debut
###############################################################################
HEIGHT = 300;
WIDTH = 600;
def plot_stacked_bar(df):
    """
    ERREUR ==> a REPRENDRE
    faire un diagramme en bar stratifies avec en abscisse les etats.
    """
    # output to static HTML file
    output_file("state_node_clique_covers_dashboard.html");
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select";
    
    states = ['etat0', 'etat1', 'etat2', 'etat3', 'etat_1'];
    data = df[states]
    data.index.name = 'graphs';
    data_gr = data.groupby('graphs')[states].sum()
    source = ColumnDataSource(data=data_gr)
    graphs = source.data['graphs'].tolist() 
    p = figure(x_range=graphs, tools=TOOLS, plot_height=HEIGHT, plot_width=WIDTH)
    
    p.vbar_stack(stackers=states, x='graphs', width=0.4,  color=Spectral5,
                 legend=states)
    p.xaxis.major_label_orientation = 1
    show(p)
    pass        
###############################################################################
#                          plot in bokeh --> fin
###############################################################################
        
if __name__ == '__main__':
    start = time.time();
    
    nbre_sommets_GR = 6#10;
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
    test_update_edges_neighbor(graphes_GR_LG);
    df_cliq_covers = test_algo_covers(graphes_GR_LG)
    #test_execute_algos(graphes_GR_LG)
    print("runtime : {}".format( time.time() - start))