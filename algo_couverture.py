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
            
            bool_clique, bool_coherent, C1, C2 = partitionner(
                                                    sommet,
                                                    sommets_k_alpha,
                                                    aretes_LG_k_alpha,
                                                    DBG
                                                    )
            
            if not bool_clique and not bool_coherent:
                sommet.etat = -1;
            else:
                if sommet.etat == 3 and len(C2) != 0:
                    sommet.etat = -1;
                    sommet.cliques_S_1 = sommet.cliques_S_1 + 2; # TODO PROBLEME remplacer 2 par le nombre de cliques trouves
                    
            pass # pass while 
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