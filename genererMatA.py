#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 12:46:57 2019

@author: willy
"""

import numpy as np
import pandas as pd
import time
import fonctions_auxiliaires as fct_aux;
import networkx as nx;
import random as rd
import random;
import math;

def algo_Welsh_Powel(matA):
    """
    but: coloration des sommets du graphe tel que 
        * 2 arcs adjacents ont des couleurs differentes
        
        particularite de existe_node_adj_in_liste_Node_NonAdj:
        recupere les noeuds n'etant pas adjacents a un "noeud defini" dans une liste "ma_liste"
        ensuite cherche les noeuds dans "ma_liste" etant adjacent entre eux et les inserer dans "ma_liste_bis"
        si il existe des noeuds dans "ma_liste_bis" alors un de ces noeuds a la meme couleur que le "noeud defini"
        et les autres autres noeuds auront des numero de couleurs differentes
    """
    liste_noeuds = matA.columns.tolist()
    liste_arcs_ = fct_aux.aretes(matA, orientation=True)
    
    # 1 liste des noeuds par ordre decroissant
    # degre de chaque noeud
    dico_degre_noeud = dict()
    for noeud in liste_noeuds:
        dico_degre_noeud[noeud] = fct_aux.degre_noeud(liste_arcs_, noeud)
    # classer liste_noeuds par ordre decroissant
    sorted_tuple_ordre_decroissant = sorted(dico_degre_noeud.items(), key=lambda x: x[1], reverse = True)
    liste_noeuds_decroissant = list()
    for tuple_ in sorted_tuple_ordre_decroissant:
        liste_noeuds_decroissant.append(tuple_[0])
    
    # 2 attribution de couleurs aux noeuds
    color = 0
    dico_color_noeud = dict()
    liste_noeuds_decroissant_copy = liste_noeuds_decroissant.copy()
    for noeud in liste_noeuds_decroissant_copy:
        # initialisation node color s
        dico_color_noeud[noeud] = None

    while liste_noeuds_decroissant_copy:
        noeud = liste_noeuds_decroissant_copy.pop()
        if dico_color_noeud[noeud] == None:
            dico_color_noeud[noeud] = color
            liste_node_nonAdj = liste_node_NonAdj_NonColorie(noeud, matA, dico_color_noeud)
            
            liste_nodes_u_v = existe_node_adj_in_liste_Node_NonAdj(list(liste_node_nonAdj), liste_arcs_)
            if len(liste_nodes_u_v)!= 0:
                nodeAdjANoeud = liste_node_nonAdj.intersection(liste_nodes_u_v)
                for node_u in nodeAdjANoeud:
                    dico_color_noeud[node_u] = color
                for u in liste_nodes_u_v:
                    dico_color_noeud[u] = color
                    color += 1
            else:
                for noeud_nonAdj in liste_node_nonAdj:
                    dico_color_noeud[noeud_nonAdj] = color
        color += 1
    
    return liste_noeuds_decroissant, dico_color_noeud 
    


def random_adjacence_matrix(size=2,nb_lien=(1,2)):
    mat = np.random.randint(0,2,(size,size))
    mat = np.tril(mat,k=-1)
    for i in range(size) :
        if i <= int(size/2) :
            l = list(mat[i+1:,i])
            while l.count(1) > nb_lien[-1] :
                indices = [i for i, x in enumerate(l) if x == 1]
                rand_index = indices[rd.randint(0,len(indices)-1)]
                l[rand_index] = 0
            while l.count(1) < nb_lien[0] :
                indices = [i for i, x in enumerate(l) if x == 0]
                rand_index = indices[rd.randint(0,len(indices)-1)]
                l[rand_index] = 1
            mat[i+1:,i] = np.asarray(l)
        else :
            l = list(mat[i,:i])
            while l.count(1) > nb_lien[-1] :
                indices = [i for i, x in enumerate(l) if x == 1]
                rand_index = indices[rd.randint(0,len(indices)-1)]
                l[rand_index] = 0
            while l.count(1) < nb_lien[0] :
                indices = [i for i, x in enumerate(l) if x == 0]
                rand_index = indices[rd.randint(0,len(indices)-1)]
                l[rand_index] = 1
            mat[i,:i] = np.asarray(l)
            
    for i in range(1,size) :
        for j in range(0,i):
            mat[j,i] = mat[i,j]
            
    return mat 
    
def genererMatriceA(dimMat, nb_aretes):
    """
    selectionne la generation de matrice matA en fonction type du nb_aretes
    nb_aretes peut etre de type:
        - tuple: on utilise genererMatriceA_nbreAreteMinMax
        - integer: on utilise generer_matrice_with_mean_degre
    """
    matA = None;
    if type(nb_aretes) == tuple:
        matA = genererMatriceA_nbreAreteMinMax(dimMat, nb_aretes);
    elif type(nb_aretes) == int:
        matA = generer_matrice_with_mean_degre(dimMat, nb_aretes);
    else:
        print("ERROR nbre d'aretes")
        return
    return matA;
    pass

def genererMatriceA_nbreAreteMinMax(dimMat, nb_lien=(2,5)):
    """
    dimMat: nombre de sommets dans le graphe
    nb_lien: le nbre d'aretes min et max
    TODO verifier sil est un graphe connexe
    """
    
    liste_noeuds =  [str(i) for i in range(dimMat)]
    matA = pd.DataFrame(columns = liste_noeuds, index = liste_noeuds)
    
    # generation graphe avec nbre de voisins min et max
    mat_ = np.random.randint(0,2,(dimMat,dimMat))
    mat_ = np.tril(mat_,k=-1)
    for i in range(dimMat) :
        if i <= int(dimMat/2) :
            l = list(mat_[i+1:,i])
            while l.count(1) > nb_lien[-1] :
                indices = [i for i, x in enumerate(l) if x == 1]
                rand_index = indices[rd.randint(0,len(indices)-1)]
                l[rand_index] = 0
            while l.count(1) < nb_lien[0] :
                indices = [i for i, x in enumerate(l) if x == 0]
                rand_index = indices[rd.randint(0,len(indices)-1)]
                l[rand_index] = 1
            mat_[i+1:,i] = np.asarray(l)
        else :
            l = list(mat_[i,:i])
            while l.count(1) > nb_lien[-1] :
                indices = [i for i, x in enumerate(l) if x == 1]
                rand_index = indices[rd.randint(0,len(indices)-1)]
                l[rand_index] = 0
            while l.count(1) < nb_lien[0] :
                indices = [i for i, x in enumerate(l) if x == 0]
                rand_index = indices[rd.randint(0,len(indices)-1)]
                l[rand_index] = 1
            mat_[i,:i] = np.asarray(l)
            
    for i in range(1,dimMat) :
        for j in range(0,i):
            mat_[j,i] = mat_[i,j]

    mat = pd.DataFrame(mat_)
    mat.columns = liste_noeuds
    mat.index = liste_noeuds
    
    # orientation des aretes 
    dico = dict()
    for noeud in liste_noeuds:
        dico[noeud] = 0
    
    liste_arcs_ = fct_aux.aretes(mat, orientation=True)
    liste_noeuds_decroissant, dico_color_noeud = algo_Welsh_Powel(mat)
    
    liste_noeuds_decroissant.reverse()    
    for noeud in liste_noeuds_decroissant:
        liste_w = fct_aux.voisins(liste_arcs_, noeud)
        for w in liste_w:
            if dico_color_noeud[noeud] < dico_color_noeud[w]:
                matA.loc[noeud][w] = 1

    matA.fillna(0,inplace = True)
    matA.index.rename("nodes",inplace=True)
    matA.loc["nodes"] = [str(i) for i in matA.index]
    return matA      
    
    
def liste_node_NonAdj_NonColorie(noeud, matA, dico_color_noeud):
    set_node_nonAdj = set()
    ind_name = matA.columns.tolist()
    for ind in ind_name:
        if ind != noeud and matA.loc[ind][noeud] == 0:
            set_node_nonAdj.add(ind)
    return set_node_nonAdj

def existe_node_adj_in_liste_Node_NonAdj(liste_node_nonAdj, liste_arcs_):
    liste_tuple = list()
    set_nodes = set()
    for u in range(len(liste_node_nonAdj)):
        for v in range(u+1,len(liste_node_nonAdj)):
            tu0 = ( liste_node_nonAdj[u], liste_node_nonAdj[v] )
            tu1 = ( liste_node_nonAdj[v], liste_node_nonAdj[u] )
            if tu0 in liste_arcs_ :
                liste_tuple.append(tu0)
                set_nodes.add( tu0[0] )
                set_nodes.add( tu0[1] )
            if tu1 in liste_arcs_:
                liste_tuple.append(tu1)
                set_nodes.add( tu1[0] )
                set_nodes.add( tu1[1] )
    return set_nodes

############### generer matrice A =====> debut ##########
def generer_matrice_with_mean_degre(dim_mat, degre_moy):
    """
    but: 
    """
    mat_ = np.random.randint(0,1,(dim_mat,dim_mat));
    proba = degre_moy/mat_.shape[0];
    ind_diagonale = 0; cpt_row = 0; index_row = 0
    for row in mat_:
        degre_row_max = math.floor(proba*mat_.shape[0]) - sum(x==1 for x in row);
        index_row = None;
        #print("*** row: ",row, " ind_diagonale: ",ind_diagonale)
        for nbre_case1 in range(0, degre_row_max):
            ind_items0 = [i[0] for i in enumerate(row) if i[1] == 0 ] 
            #print("cpt_row: ", cpt_row," degre_row_max: ",degre_row_max,\
            #      " nbre_case1: ",nbre_case1," ind_items0: ",ind_items0,\
            #      " index_row: ",index_row," ind_diagonale: ",ind_diagonale)
            if len(ind_items0) != 0:
                index_row = random.choice( ind_items0 );
                row[index_row] = 1;
                mat_[:,cpt_row][index_row] = 1
        ind_diagonale += 1; cpt_row += 1;
        #print("***(apres) row: ",row)
    np.fill_diagonal(mat_, 0)
    noeuds = [str(i) for i in range(mat_.shape[0])]
    mat = pd.DataFrame(mat_, index = noeuds, columns = noeuds);
    aretes = fct_aux.aretes(mat, orientation=True)
    
    # graphes connexes
    mat = graphe_connexe(mat, aretes);
    
    matA = orienter_graphe(mat, noeuds, aretes)
    G_ = nx.Graph( fct_aux.aretes(matA, orientation=True))
    
    matA.index.rename("nodes",inplace=True)
    matA.loc["nodes"] = [str(i) for i in matA.index]
    
#    print("matA is_directed = ", nx.is_directed_acyclic_graph(G_))
    return matA;   
            
def orienter_graphe(mat_, noeuds, aretes):
    """
    a tester sur un graphe a la main
    """
#    noeuds = [str(i) for i in range(mat.shape[0])]
#    mat_ = pd.DataFrame(mat, index = noeuds, columns = noeuds);
    matA = pd.DataFrame(0, index = noeuds, columns = noeuds);
    noeud_source = random.choice(noeuds);
#    noeud_source = "0"
    noeud_select = noeud_source;      
    dico_gamma = fct_aux.gamma_noeud_(mat_); # {"2":[3,{"1","3","4"},....]}
#    aretes = fct_aux.aretes(mat_,, orientation=True)
#    print("aretes: ",aretes)
    cpt_noeuds = 0;
    file_gamma = list(dico_gamma[noeud_source][1]);
    dico_nodes_markes = dict(zip(noeuds, [0]*len(noeuds)));  # 0: non marke, 1: marke;
    dico_nodes_color = dict(zip(noeuds, [0]*len(noeuds)));
    dico_nodes_color[noeud_select] = cpt_noeuds
#    print("noeud_source: ", noeud_source, "file_gamma:",file_gamma)
    while( len(file_gamma) != 0):
        dico_nodes_markes[noeud_select] = 1;
        noeud_voisin = file_gamma.pop(0);
        if dico_nodes_markes[noeud_voisin] == 0:
           file_gamma.extend(list(dico_gamma[noeud_voisin][1])); cpt_noeuds += 1;
           dico_nodes_color[noeud_voisin] = cpt_noeuds;
           if dico_nodes_color[noeud_select] < dico_nodes_color[noeud_voisin] and \
               (noeud_select,noeud_voisin) in aretes or (noeud_voisin,noeud_select) in aretes:
               matA.loc[noeud_select][noeud_voisin] = 1;
           #else:
           #    matA.loc[noeud_voisin][noeud_select] = 1;
           elif dico_nodes_color[noeud_select] >= dico_nodes_color[noeud_voisin] and \
               (noeud_select,noeud_voisin) in aretes or (noeud_voisin,noeud_select) in aretes:
               matA.loc[noeud_voisin][noeud_select] = 1;
               
           if (noeud_select,noeud_voisin) in aretes:
               aretes.remove((noeud_select,noeud_voisin))
           if (noeud_voisin,noeud_select) in aretes:
               aretes.remove((noeud_voisin,noeud_select))   
        noeud_select = noeud_voisin;
    while(len(aretes) != 0):
        arete = aretes.pop()
        if dico_nodes_color[arete[0]] < dico_nodes_color[arete[1]]:
            matA.loc[arete[0]][arete[1]] = 1;
        else:
            matA.loc[arete[1]][arete[0]] = 1;
            
        
    return matA
############### generer matrice A =====> fin ############

#### connected components #####
def graphe_connexe(mat_, aretes):
#    aretes = fct_aux.aretes(mat_, orientation=True);
    G = nx.Graph(aretes);
#    print("mat_ is_DAG = ", nx.is_directed(G), " is_connected = ",nx.is_connected(G))
    if nx.is_connected(G):
        return mat_;
    else:
        components = list(nx.connected_components(G))
        for ind_i in range(len(components)-1):
            for ind_j in range(ind_i, len(components)):
                node_1 = random.choice(list( components[ind_i] ));
                node_2 = random.choice(list( components[ind_j] ));
                mat_.loc[node_1][node_2] = 1
                mat_.loc[node_2][node_1] = 1
        G_ = nx.Graph( fct_aux.aretes(mat_, orientation=True))
#        print("mat_ is_connected = ", nx.is_connected(G_))
    return mat_    
    pass
#### connected components #####
    
if __name__ == "__main__" :
    
    start= time.time()
    dim_mat = 6; degre_moy = 3;
#    a= np.array([[0,1,0,0,0,0],[1,0,1,0,0,0],[0,1,0,0,0,0],[0,0,0,0,1,1],[0,0,0,1,0,1],[0,0,0,0,1,0]])
#    mat_a = pd.DataFrame(a)
#    mat_a = connexe(mat_a)
#    mat = generer_matrice_with_mean_degre(dim_mat, degre_moy)
#    print("\n", mat)
#    matA = genererMatriceA(10,(2,5))
#    matA.to_csv("df_matA_generer.csv")
#    print(matA)
#    print (time.time() - start)
#    mat = random_adjacence_matrix(5,(1,2))
#    a = np.random.randint(0,2,(5,5));
#    m_ = np.tril(a) + np.tril(a, -1).T;
#    no = [str(i) for i in range(m_.shape[0])];
#    mat = pd.DataFrame(m_, index = no, columns = no)
#    matA = orienter_graphe(mat)

#    g_test = np.array([[0,1,1,0,0,0],[1,0,1,1,0,0],[1,1,0,1,1,0],\
#                       [0,1,1,0,1,1],[0,0,1,1,0,1],[0,0,0,1,1,0]], np.int32)
#    mat_g = orienter_graphe(g_test);
    print("\n")
#    print(mat_)