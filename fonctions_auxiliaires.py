#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 00:44:57 2019

@author: willy
"""
import pandas as pd;
import itertools as it;

def aretes(mat_GR, orientation = False, val_0_1 = 1) :
    """ methode qui retourne soit les arcs de graphe si orientation = True
        ou les arete si orientation = False
    """
    aretes_arcs = set();
    if orientation :                                                            # True = arcs
        if isinstance(mat_GR, pd.DataFrame) :
            for (u,v) in it.permutations(mat_GR.columns, 2) :
                if mat_GR.loc[u,v] == val_0_1 : #or mat_GR.loc[v,u] == 1 :
                    aretes_arcs.add((u,v))
        elif isinstance(mat_GR, dict) :
            pass
    else :                                                                      # False = arete
        if isinstance(mat_GR, pd.DataFrame) :
            for (u,v) in it.permutations(mat_GR.columns, 2) :
                if mat_GR.loc[u,v] == val_0_1 or mat_GR.loc[v,u] == val_0_1 :
                    aretes_arcs.add( frozenset((u,v)) )
        elif isinstance(mat_GR, dict) :
            pass
        
    return aretes_arcs;

def degre_noeud(aretes, noeud):
    """
    retourne le nbre d arcs ayant un noeud en commun 
    """
    cpt = 0
    for arc in aretes:
        if noeud == arc[0] or noeud == arc[1]:
           cpt += 1 
    return cpt

def voisins(aretes, noeud):
    """
    recherche pour chaque arc si noeud est une extremite de cet arc.
    La  variable aretes est soit :
        * la liste des aretes du mat_LG
        * la matrice mat_LG
    """        
    liste_voisins = list()
    if isinstance(aretes, set):
        for arc in aretes:
            arc = list(arc)
            if noeud == arc[0]:
                liste_voisins.append( arc[1] )
            if noeud == arc[1]:
                liste_voisins.append( arc[0] )
    elif isinstance(aretes, pd.DataFrame):
        for sommet_adjacent_possible in aretes[:noeud]:
            if aretes.loc[sommet_adjacent_possible, noeud] == 1 :
                liste_voisins.append(sommet_adjacent_possible);
    return liste_voisins

def convert_sommet_to_df(sommets_k_alpha):
    """
    convertir le dictionnaire contenant des objets de type Noeud en un
    DataFrame.
    """
    som_cols = list(sommets_k_alpha.keys())
    df = pd.DataFrame(columns = som_cols, index = som_cols);
    for nom_sommet, sommet in sommets_k_alpha.items():
        for voisin in sommet.voisins:
            df.loc[nom_sommet, voisin] = 1;
            df.loc[voisin, nom_sommet] = 1;
     
    df.fillna(0, inplace=True);
    return df.astype(int);


def is_exists_sommet(sommets, etat_1=-1):
    """
    verifier s'il existe au moins un sommet ayant un etat a -1. 
    """
    bool_is_exists = False;
    nom_soms = list(sommets.keys())
    while not bool_is_exists and len(nom_soms) != 0:
        nom_som = nom_soms.pop()
        if sommets[nom_som].etat == etat_1:
            bool_is_exists = True;
    
        pass # end while not 
    return bool_is_exists;
    pass

def modify_state_sommets_mat_LG(sommets,
                          sommets_res):
    """
    modifier les etats de mat_LG en fpnction du resultat des algos contenu 
    dans sommets_k_alpha_res.
    """
    for nom_som, sommet in sommets.items():
        sommet.etat = sommets_res[nom_som].etat;
        sommet.cliques_S_1 = sommets_res[nom_som].cliques_S_1;
    
    return sommets
    pass

def node_names_by_state(sommets, etat_1):
    """
    retourner le nom des sommets ayant pour etat etat_1.
    """
    noms_sommets = set()
    for nom, sommet in sommets.items():
        if sommet.etat == etat_1:
            noms_sommets.add(nom);
            
    return noms_sommets;

def grouped_cliques_by_node(cliques, noms_sommets_1):
    """
    retourner un dictionnaire contenant les cliques par sommets
    """
    dico = dict(); 
    dico = dict.fromkeys(noms_sommets_1,[]);
    for clique  in cliques:
        sommets_communs = clique.intersection(noms_sommets_1);
        for nom_sommet in sommets_communs:
            if nom_sommet not in dico:
                dico[nom_sommet] = [clique]
            else:
                dico[nom_sommet].append(clique)
                
    return dico;

def edges_in_cliques(cliques_couvertures):
    """ retourne les aretes de tous les cliques. """
    aretes_cliques = list();
    
    boolean_subset = False;
    for clique in cliques_couvertures:
        if isinstance(clique, list) or \
            isinstance(clique, set) or \
            isinstance(clique, frozenset):
            boolean_subset = True;
            
    if boolean_subset:
        aretes_cliques = [frozenset(item) 
                            for sublist in [list(it.combinations(clique,2)) 
                                            for clique in cliques_couvertures] 
                            for item in sublist]
    else:
        aretes_cliques = list(it.combinations(cliques_couvertures,2));
    return aretes_cliques;