#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:57:48 2019

@author: willy
"""

import os;
import time;
import math;
import random;
import numpy as np;
import pandas as pd;
import itertools as it;
import networkx as nx;
import clique_max as clique;
import fonctions_auxiliaires as fct_aux;


###############################################################################
#               graphes doubles -> DEBUT
#            verify if mat_LG_k_alpha is isomorphic to one graph in GRAPHE_DOUBLE
###############################################################################
TRIANGLE = { frozenset({'A','B'}),frozenset({'A','C'}),frozenset({'B','C'}) }
CERVOLANT = { frozenset({'A','B'}), frozenset({'A','C'}),
              frozenset({'B','C'}),frozenset({'C','D'}),frozenset({'B','D'}) }
LOSANGE = { frozenset({'A','B'}), frozenset({'A','C'}), frozenset({'A','E'}),
            frozenset({'B','E'}), frozenset({'E','C'}), frozenset({'B','D'}),
            frozenset({'E','D'}), frozenset({'C','D'}) }
VOILE = { frozenset({'A','B'}), frozenset({'A','E'}), frozenset({'A','F'}), 
          frozenset({'A','C'}), frozenset({'B','C'}), frozenset({'C','F'}),
          frozenset({'F','E'}), frozenset({'B','E'}), frozenset({'C','D'}),
          frozenset({'F','D'}), frozenset({'E','D'}), frozenset({'B','D'}) }
GRAPHES_DOUBLES = {
        "triangle":TRIANGLE,
        "cervolant":CERVOLANT,
        "losange": LOSANGE,
        "voile": VOILE
        }
def is_isomorphic(aretes_LG_k_alpha):
    """
    return True if mat_LG_k_alpha(aretes_LG_k_alpha) est isomorphe 
    a l'un des graphes de GRAPHES_DOUBLES.
    """
    G_k_alpha = nx.Graph(); G_k_alpha.add_edges_from(aretes_LG_k_alpha)
    
    bool_isomorphe = False;
    for nom_graph, aretes_graph in GRAPHES_DOUBLES.items():
        G = nx.Graph();
        G.add_edges_from(aretes_graph);
        if nx.is_isomorphic(G_k_alpha, G):
            bool_isomorphe == True
            return True;
        
    return False;
    
###############################################################################
#               graphes doubles -> FIN
###############################################################################

###############################################################################
#              caracteristiques d'un sommet -> DEBUT
#               is_state, selected_node
###############################################################################
def is_state(sommets_k_alpha, critere0=0, critere3=3):
    """
    return True s'il existe des sommets dont leur etat = 0 ou 3.
    """
    bool_is_state = False;
    nom_sommets = list(sommets_k_alpha.keys());
    
    while not bool_is_state and len(nom_sommets) != 0:
        id_nom_som, nom_sommet = random.choice(list(enumerate(nom_sommets)))
        nom_sommets.pop(id_nom_som);
        sommet = sommets_k_alpha[nom_sommet];
        if sommet.etat == critere0 or sommet.etat == critere3:
            bool_is_state = True;
            #print("nom={},etat={}".format(sommet.nom, sommet.etat))
        pass # while 
    return bool_is_state;

def selected_node(sommets_k_alpha, critere0=0, critere3=3):
    """
    retourner un sommet ayant un etat egal a critere0 sinon egal a critere3.
    Dans le cas ou il n'existe aucun sommet dont l etat est egal a critere0 ou 
    critere3 alors la fonction retourne None
    """
    selected_sommet = None;
    nom_sommets = list(sommets_k_alpha.keys());
    critere03 = critere0;
    treated_yet_0 = True;
    
    while selected_sommet == None and len(nom_sommets) != 0:
        id_nom_som, nom_sommet = random.choice(list(enumerate(nom_sommets)))
        nom_sommets.pop(id_nom_som);
        sommet = sommets_k_alpha[nom_sommet];
        if sommet.etat == critere03:
            selected_sommet = sommet;
            
        if len(nom_sommets) == 0 and treated_yet_0:
            treated_yet_0 = False; 
            critere03 = critere3;
            nom_sommets = list(sommets_k_alpha.keys());
        pass #while
        
    return selected_sommet;
    pass
###############################################################################
#              caracteristiques d'un sommet -> FIN
###############################################################################

###############################################################################
#              Partitionner un sommet et son voisinage en cliques -> DEBUT
###############################################################################
def build_matrice_of_subgraph(sommet, sommets_k_alpha):
    """
    construire une matrice du sous-graphe d'un sommet et de son voisinage.
    """

    voisins = set(sommets_k_alpha[sommet].voisins);
        
    noeuds_subgraph = voisins.copy()
    noeuds_subgraph.add(sommet);
    mat_subgraph = pd.DataFrame(columns=noeuds_subgraph, index=noeuds_subgraph);
    
    for noeud in voisins:
        voisins_noeud = set(sommets_k_alpha[noeud].voisins)
        noeuds_int = noeuds_subgraph.intersection(voisins_noeud);
        for noeud_int in noeuds_int:
            mat_subgraph.loc[noeud, noeud_int] = 1;
            mat_subgraph.loc[noeud_int, noeud] = 1;
            
    return mat_subgraph.fillna(0).astype(int);
    pass

def partitionner(sommet,
                 sommets_k_alpha,
                 aretes_LG_k_alpha,
                 DBG
                 ):
    """
    retourner la liste des cliques couvrant un sommet et son voisinage.
    """
    sommet = set(sommets_k_alpha[sommet].voisins);
    
    mat_subgraph = build_matrice_of_subgraph(sommet=sommet, 
                                             sommets_k_alpha=sommets_k_alpha)
    voisins = set(sommets_k_alpha[sommet].voisins);
    voisins.add(sommet)
    cliques = clique.find_clique(mat_subgraph, 
                                 voisins, 
                                 [])
    cliques = [set(c) for c in cliques if sommet in c];
    return cliques
    pass
###############################################################################
#              Partitionner un sommet et son voisinage en cliques -> FIN
###############################################################################

###############################################################################
#               verifier la coherence des cliques -> DEBUT
###############################################################################
def verify_cliques(cliques, nom_sommet):
    """
    verifier si les cliques sont coherentes cad les sommets sont une extremite commune.
    Si non chercher les cliques coherentes.
    cliques_coh = cliques coherentes
    """
    f=lambda x: set(x.split("_"))
    
    cliques_coh, cliques_coh_tmp = [],[];
    cliques_a_frag = []
    for cliq in cliques:
        aretes = []
        aretes = list(map(f, cliq))
        sommet_commun = None;
        sommet_commun = set.intersection(*aretes);
        if sommet_commun != None and len(sommet_commun) == 1:
            cliques_coh_tmp.append(cliq)
        else:
            cliques_a_frag.append(cliq)
    
#    print("cliques_coh_tmp={}, cliques_a_frag={}".format(cliques_coh_tmp, 
#                                              cliques_a_frag))
    
    for cliq_a_frag in cliques_a_frag:
        if len(cliq_a_frag) == 3:
            sub_sets = list(it.combinations(cliq_a_frag,2));
            sub_sets = [set(s) for s in sub_sets if nom_sommet in s]
#            print("cliq_a_frag={},sub_sets={}".format(cliq_a_frag, sub_sets))
            
            for sub_set in sub_sets:
                bool_subset = True;
                for cliq_coh_tmp in cliques_coh_tmp:
                    if sub_set.issubset(cliq_coh_tmp) or \
                        cliq_coh_tmp.issubset(sub_set) :
                        bool_subset = False;
#                print("***{} subset of one of {}? {}".format(sub_set,cliques_coh_tmp, bool_subset))
                if bool_subset:
                    cliques_coh_tmp.append(sub_set);
            
    bool_clique, bool_coherent, cliques_coh = False, False, []

    if len(cliques_coh_tmp) == 1:
        cliques_coh = cliques_coh_tmp.copy()
        bool_clique, bool_coherent = True, True;
    elif len(cliques_coh_tmp) == 2:
        bool_clique, bool_coherent = True, True;
        cliques_coh = cliques_coh_tmp.copy();
    else:
        cliques_coh = cliques_coh_tmp.copy();
        
    return bool_clique, bool_coherent, cliques_coh;
    pass
###############################################################################
#               verifier la coherence des cliques -> FIN
###############################################################################

###############################################################################
#               supprimer les aretes des cliques C1, C2 et 
#                   mettre a jour l etat des sommets -> DEBUT
###############################################################################
def update_edges_neighbor(C1, C2, aretes, sommets):
    """
    supprimer les aretes des cliques C1, C2 dans la mat_LG et 
    mettre a jour le voisinage de chaque sommet.
    
    aretes = ce sont les aretes de mat_LG_k_alpha cad aretes_LG_k_alpha
    sommets = dictionnaire de classe sommet cad sommets_k_alpha
    """
    
    pass
###############################################################################
#               supprimer les aretes des cliques et 
#                   mettre a jour l etat des sommets -> FIN
###############################################################################

###############################################################################
#               cover algorithm -> DEBUT
###############################################################################
def clique_covers(mat_LG_k_alpha, 
                aretes_LG_k_alpha, 
                sommets_k_alpha,
                DBG):
    """
    couvrir chaque arete de mat_LG_k_alpha par une clique.
    """
    cliques_couvertures = set();
    
    if is_isomorphic(aretes_LG_k_alpha):
        print("attention A revoir APRES")
    else :
        while is_state(sommets_k_alpha, critere0=0, critere3=3):
            sommet = selected_node(sommets_k_alpha, 
                                   critere0=0, 
                                   critere3=3)
            
            if sommet == None:
                return cliques_couvertures, sommets_k_alpha;
            
            cliques = partitionner(
                                sommet,
                                sommets_k_alpha,
                                aretes_LG_k_alpha,
                                DBG
                                )
            
            cliques_coh = []
            bool_clique, bool_coherent, cliques_coh = \
                                verify_cliques(
                                    cliques = cliques,
                                    nom_sommet = sommet.nom)
            C1, C2 = set(), set(); 
            if len(cliques_coh) == 1:
                C1 = cliques_coh[0];
            elif len(cliques_coh) == 2:
                C1, C2 = cliques_coh[0], cliques_coh[1];
                
                
            if not bool_clique and not bool_coherent:
                sommet.etat = -1;
                sommet.cliques_S_1 = sommet.cliques_S_1 + len(cliques_coh);
            else:
                if sommet.etat == 3 and len(C2) != 0:
                    sommet.etat = -1;
                    sommet.cliques_S_1 = sommet.cliques_S_1 + 2; 
                
                for voisin_som in sommet.voisins:
                    if len(C1.union(C2).union(sommets_k_alpha[voisin_som]) - \
                        C1.union(C2)) != 0:
                        if sommets_k_alpha[voisin_som].etat == 3:
                            sommets_k_alpha[voisin_som].etat = -1;
                        elif sommets_k_alpha[voisin_som].etat == 0:
                            sommets_k_alpha[voisin_som].etat = 3;
                    else:
                        if sommets_k_alpha[voisin_som].etat == 3:
                            sommets_k_alpha[voisin_som].etat = 2;
                        elif sommets_k_alpha[voisin_som].etat == 0:
                            sommets_k_alpha[voisin_som].etat = 1;
                    sommets_k_alpha[voisin_som].cliques_S_1 += 1;
                    pass # pass for voisin_som
                
                # mise a jour de l etat du sommet.
                if sommet.etat == 0:
                    if len(C2) == 0:
                        sommet.etat = 1;
                        sommet.cliques_S_1 = sommet.cliques_S_1 + 1; 
                    else:
                        sommet.etat = 2;
                        sommet.cliques_S_1 = sommet.cliques_S_1 + 2; 
                else:
                    sommet.etat = 2;
                    sommet.cliques_S_1 = sommet.cliques_S_1 + 1; 
                    
                # suppression aretes des cliques  et mise a jour du voisinage des sommets
                aretes_LG_k_alpha, sommets_k_alpha = update_edges_neighbor(
                                                    C1 = C1,
                                                    C2 = C2,
                                                    aretes = aretes_LG_k_alpha,
                                                    sommets = sommets_k_alpha)
                
                for C in [C1,C2]:
                    if C :
                        cliques_couvertures.add(frozenset(C))
            pass # pass while
            
        print("cliques : {}, aretes = {}".format(len(cliques_couvertures), 
                                                  len(aretes_LG_k_alpha)))
        
        return cliques_couvertures, aretes_LG_k_alpha, sommets_k_alpha;
    pass
###############################################################################
#               cover algorithm -> FIN
###############################################################################

###############################################################################
#               -> DEBUT
###############################################################################
###############################################################################
#               -> FIN
###############################################################################
if __name__ == '__main__':
    start = time.time();
    
    print("runtime : {}".format( time.time() - start))