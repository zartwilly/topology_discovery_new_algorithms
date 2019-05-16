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