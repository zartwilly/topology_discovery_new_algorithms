#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 00:28:39 2019

@author: willy
"""
import time;
import pandas as pd
import itertools as it;
import genererMatA as geneMatA;
import fonctions_auxiliaires as fct_aux;

import defs_classes as def_class;

from pathlib import Path;


INDEX_COL_MATE_LG = "sommets_aretes";
INDEX_COL_MAT_GR = "nodes";
NOM_MATE_LG = "matE_generer.csv";
NOM_MAT_GR = "mat_generer.csv";


def creer_mat_LG(arcs_or_aretes) :
    """ Methode qui determine la matrice de line graphe de mat_GR a partir de
        la liste des arcs ou aretes.
    """
    mat_LG = None;
    
#    aretes = list(map(set, arcs_or_aretes))
    aretes = map(set, arcs_or_aretes)
    
    dico_graphe = dict()
    for (arete0,arete1) in it.combinations(aretes,2) :
        if arete0.intersection(arete1) :
            arete0 = "_".join(sorted(arete0))
            arete1 = "_".join(sorted(arete1))
            if arete0 not in dico_graphe and arete1 not in dico_graphe :
                dico_graphe[arete0] = [arete1];
            elif arete0 not in dico_graphe and arete1 in dico_graphe :
                dico_graphe[arete1].append(arete0);
            elif arete0 in dico_graphe and arete1 not in dico_graphe :
                dico_graphe[arete0].append(arete1);
            elif arete0 in dico_graphe and arete1 in dico_graphe :
                dico_graphe[arete0].append(arete1);
                
    mat_LG = pd.DataFrame(index = dico_graphe.keys(), 
                          columns = dico_graphe.keys());
    for k, vals in dico_graphe.items() :
        for v in vals:
            mat_LG.loc[k,v] = 1
            mat_LG.loc[v,k] = 1
    mat_LG.fillna(value=0, inplace=True);
    
    return mat_LG.astype(int);

def generer_reseau(nbre_sommets_GR, nbre_moyen_liens, chemin_matrice):
    """ Methode que cree le graphe racine et son line graphe.
    """
    mat_GR = geneMatA.genererMatriceA(nbre_sommets_GR, nbre_moyen_liens);
    arcs = fct_aux.aretes(mat_GR, orientation = True);
    mat_LG = creer_mat_LG(arcs);
    
    path_matrice = Path(chemin_matrice);
    if not path_matrice.is_dir() :
        path_matrice.mkdir(parents=True, exist_ok=True)
        
    mat_LG.to_csv(chemin_matrice + NOM_MATE_LG, 
                index_label = INDEX_COL_MATE_LG)
    mat_GR.to_csv(chemin_matrice + NOM_MAT_GR)
    return mat_LG, mat_GR;

def sommets_mat_LG(mat_LG, etat=0):
    """
    obtenir la liste des sommets sous forme de classes
    """
    sommets = dict();
    for nom_sommet in mat_LG.columns :
        sommet = def_class.Noeud(nom_sommet, etat);
        voisins = fct_aux.voisins(mat_LG, nom_sommet);
        sommet.voisins = frozenset(voisins);
        sommet.ext_init = nom_sommet.split("_")[0];
        sommet.ext_final = nom_sommet.split("_")[1];
        sommets[nom_sommet] = sommet;
        
    return sommets;
      
def creer_graphe(nbre_sommets_GR = 5, 
                 nbre_moyen_liens = (2,5), 
                 chemin_matrice=""):
    """
    creation d'un graphe GR et son line graphe LG.
    retourne mat_GR, mat_LG, sommets, aretes.
    """
    mat_LG, mat_GR = generer_reseau(nbre_sommets_GR, 
                                    nbre_moyen_liens, 
                                    chemin_matrice)
    sommets = sommets_mat_LG(mat_LG);
    aretes = fct_aux.aretes(mat_LG, orientation=False)
    return mat_GR, mat_LG, sommets, aretes;


if __name__ == '__main__':
    ti = time.time();
    
    log_file = "DEBUG.log";
    
    NBRE_GRAPHES = 10#300; #1
    rep_data = "data_test"
    dbg = False#True;
    
    
    # caracteristiques graphes racines
    nbre_sommets_GR = 5#8#5;
    nbre_moyen_liens = (2,5);
    epsilon = 0.75; effet_joule = 0;
    nbre_ts = 10;
    chemin_matrice = rep_data + "/" + "matrices" + "/";
    
    mat_LG, mat_GR = generer_reseau(nbre_sommets_GR, 
                                    nbre_moyen_liens, 
                                    chemin_matrice)
    
     
    # creer les sommets avec les classes
    etat = 0
    sommets = sommets_mat_LG(mat_LG, etat);
    
    #TODO
    # executer avec spark
    # injecter dans elasticSearch dans un shark