#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 06:23:38 2019

@author: willy
"""
import os, sys;
import pandas as pd;

def lire_k_erreurs(rep_dist):
    """
    lire les k_erreurs dans les distributions
    """
   
    k_erreurs = [int(dist.split("_")[4].split(".")[0]) 
                    for dist in os.listdir(rep_dist) 
                    if int(dist.split("_")[4].split(".")[0]) != 0]
    
    return k_erreurs;
    
def create_dataframe_data_revue(rep_dist, k_erreurs,
                                    DISTRIB_ROOT_FILE,
                                    DISTRIB_EXT_FILE, 
                                    NAMES_HEADERS):
    """
    creer le dataframe, issue des donnees pour la revue scientifique 
    contenant les caracteristiques des k_erreurs.
    
    num_graph = G_numeroGraphe_kerreur_p_Pcorrel #exple:G_3_5_p_05
    """
    
    f = lambda row: "_".join([row["num_graph"].split("_")[0], 
                              row["num_graph"].split("_")[1],
                                row["num_graph"].split("_")[2]
                                ])
    # X1 = abs(moy_distline - k);
    # correl_dl_dh = abs(moy_hamming - X1) / (k + moy_distline);
    f_cal_correl = lambda row: (abs(row["dh"] \
                                    - abs(row['dc'] \
                                          -row['k_erreur']
                                          ) 
                                    )) / (row['k_erreur'] + row['dc'])
    
    frames = []
    for k_erreur in k_erreurs:
        df_k = pd.read_csv(
                    rep_dist+DISTRIB_ROOT_FILE+str(k_erreur)+DISTRIB_EXT_FILE,
                    names=NAMES_HEADERS,
                    sep=";"
                         );
        frames.append(df_k);
    
    df = pd.DataFrame();
    df = pd.concat(frames, ignore_index=True)
    df["num_graph"] = df.apply(f, axis=1);
    df["correl_dc_dh"] = df.apply(f_cal_correl, axis=1);
    return df;

def title_xlabel_ylabel_figure(dc_dh_correl, k_erreur, mu, sigma, BOOL_ANGLAIS):
    """
    retourner le titre, le label de l axe x et celui de l axe y.
    """
    title, xlabel, ylabel = "", "", "";
    if BOOL_ANGLAIS:
        if dc_dh_correl == "moy_dc":
            title = "DC for k={}".format(k_erreur);
            xlabel = "DC";
            ylabel = "graph_number (in %)";
        elif dc_dh_correl == "moy_dh":
            title = "DH for k={}".format(k_erreur);
            xlabel = "DH";
            ylabel = "graph_number (in %)";
        elif dc_dh_correl == "correl":
            title = "cumulative correlation \n between DC and DH \n"\
                    +"for k={} added/deleted edges".format(k_erreur);
            title = "cumulative correlation \n between DC and DH \n"\
                    +"for k={}".format(k_erreur);
            xlabel = "correlation_DC_DH";
            ylabel = "cumulative correlation";
        elif dc_dh_correl == "cumul_correl":
            title = "cumulative correlation for \n"\
                    +str(k_erreur)\
                    +"  added/deleted edges";
            xlabel = "correl_DC_DH";
            title = "cumulative correlation \nfor k="\
                    +str(k_erreur);
            xlabel = "correl";
            ylabel = "graph_number_correl_DC_DH<x";
        elif dc_dh_correl == "cumul_dh":
            title = "cumulative DH for \n"\
                    +str(k_erreur)\
                    +"  added/deleted edges";
            title = "cumulative DH for k="\
                    +str(k_erreur);
            xlabel = "DH";
            ylabel = "graph_number_DH<x";
        
    else:
        if dc_dh_correl == "moy_dc":
            title = "DC pour k={}".format(k_erreur);
            xlabel = "DC";
            ylabel = "nombre_graphe (en %)";
        elif dc_dh_correl == "moy_dh":
            title = "DH pour k={}".format(k_erreur)
            xlabel = "DH";
            ylabel = "nombre_graphe (en %)";
        elif dc_dh_correl == "correl":
            title = "correlation cumulative entre DC et DH \n"\
                    +"pour k={}".format(k_erreur);
            xlabel = "correlation_DC_DH";
            ylabel = "correlation cumulative";
        elif dc_dh_correl == "cumul_correl":
            title = "correlation cumulative \npour k="\
                    +str(k_erreur);
            xlabel = "correl";
            ylabel = "nb_graphe_correl_DC_DH<x";
        elif dc_dh_correl == "cumul_dh":
            title = "DH cumulative pour k="\
                    +str(k_erreur);
            xlabel = "DH";
            ylabel = "nb_graphe_DH<x";
        
        
    return title, xlabel, ylabel;