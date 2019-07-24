#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:56:31 2019

@author: willy
"""

import time;
import itertools as it;
import bokeh_revue as bkh;
import matplotlib_revue as matplib;

import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;
import itertools as it;
import fonctions_auxiliaires_visu as fct_aux_vis;
import matplotlib.font_manager as font_manager;

from pathlib import Path;
from scipy.stats import norm;
from matplotlib import lines, markers;

from bokeh.plotting import *
from bokeh.layouts import grid, column;
from bokeh.models import ColumnDataSource, FactorRange;
from bokeh.plotting import figure, output_file, show, gridplot;
from bokeh.core.properties import value
from bokeh.palettes import Spectral5
from bokeh.models.tools import HoverTool
from bokeh.models.tickers import FixedTicker
from bokeh.models import FuncTickFormatter
###############################################################################
#                       CONSTANTES  ===> debut
###############################################################################
NAMES_HEADERS = ["num_graph", "k_erreur", "alpha", "dc", "dh", 
                     "aretes_matE", "correl_dc_dh", "runtime"];
NAME_COURBES = ["DC","DH","CUMUL_CORREL","CUMUL_DH"];
DISTRIB_ROOT_FILE = "distribution_moyDistLine_moyHamming_k_";
DISTRIB_EXT_FILE = ".txt";
DATA_P_REP = "data_p_";

WIDTH = 400;
HEIGHT = 400;
MUL_WIDTH = 2.5;
MUL_HEIGHT = 3.5;

MARKERS = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", 
               "P", "*", "h", "H", "+", "x", "X", "D", "d"]

BOOL_ANGLAIS = True;
###############################################################################
#                       CONSTANTES  ===> fin
###############################################################################

###############################################################################
#           representation du tps d'execution des k_erreurs => debut
###############################################################################
def plot_bokeh_tps_calcul_k_1(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS):
    """
    representer le temps d'execution pour chaque k_erreur
    """
    colors = ["red", "yellow", "blue", "green", "rosybrown", 
              "darkorange", "fuchsia", "grey"]
    
    df_grouped_numgraph = df_kerrs.groupby("num_graph")\
                                ["dc",'dh','k_erreur','correl_dc_dh','runtime'].mean()
    
    title = "execution times" + " (a)"
    xlabel = "graph number"
    ylabel = "seconds"
    p_cal = figure(plot_height = int(HEIGHT * facteur_height), 
                    plot_width = int(WIDTH * facteur_width), 
                    title = title,
                    x_axis_label = xlabel, 
                    y_axis_label = ylabel, 
                    tools = TOOLS)
                            
    for ind_k_erreur, k_erreur in enumerate(k_erreurs):
        if k_erreur == 30 :
            continue;
        df_grouped_k = df_grouped_numgraph[ 
                            df_grouped_numgraph["k_erreur"] == k_erreur ]
        if k_erreur == 1:
            df_grouped_k = df_grouped_k[df_grouped_k['dc'] == 1]
            
        df_grouped_k = df_grouped_k.sort_values(by='runtime', axis=0)
        
        df_grouped_k['graph_number'] = np.arange(1, df_grouped_k.shape[0]+1);
        
        src = ColumnDataSource(df_grouped_k);
        
        p_cal.line(x="graph_number", y="runtime", source=src, 
                   color=colors[ind_k_erreur],
                   legend= "k = "+str(k_erreur),
                   line_width=2, line_dash="dashed");
    
    p_cal.legend.location = "top_left"
    return p_cal
###############################################################################
#           representation du tps d'execution des k_erreurs => fin
###############################################################################
    
###############################################################################
#       representation du diagramme en baton des DH en fonction des k_erreurs => debut
###############################################################################
def plot_bokeh_baton_dh_k_1(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS):
    """
    representation du diagramme en baton des DH en fonction des k_erreurs.
    """
    df_grouped_numgraph = df_kerrs.groupby("num_graph")\
                                ["dc",'dh','k_erreur','correl_dc_dh','runtime'].mean()
    
    df_grouped = pd.DataFrame()
    frames = []
    for k_erreur in k_erreurs:
        df_grouped_k = df_grouped_numgraph[ 
                            df_grouped_numgraph["k_erreur"] == k_erreur ]
        if k_erreur == 1:
            df_grouped_k = df_grouped_k[df_grouped_k['dc'] == 1]
            
        df_gr_k_sorted = df_grouped_k.sort_values(by=['k_erreur','dh'], axis=0);
        df_gr_k_sorted['numero_graphe'] = np.arange(1, df_gr_k_sorted.shape[0]+1)
        frames.append(df_gr_k_sorted)
        
    df_grouped = pd.concat(frames)
    df_grouped["k_erreur"] = df_grouped["k_erreur"].astype(int)
    df_grouped["dh"] = df_grouped["dh"].astype(int)
    
    subset = df_grouped[["k_erreur","numero_graphe"]]
    tuples_x = [tuple((str(x[0]), str(x[1]))) for x in subset.values]
    dh = list(df_grouped["dh"].values);
    
    src = ColumnDataSource(data=dict(x=tuples_x, dh=dh))
    
    title = "Hamming distance by k added/deleted edges"
    xlabel = "k_error"
    ylabel = "DH_k"
    p_baton = figure(
                    plot_height = int(HEIGHT*facteur_height), 
                    plot_width = int(WIDTH*facteur_width),
                    x_range=FactorRange(*tuples_x),
                    title = title,
                    x_axis_label = xlabel, 
                    y_axis_label = ylabel, 
                    tools = TOOLS)
    
    p_baton.vbar(x='x', top='dh', width=0.1, source=src)

    p_baton.y_range.start = 0
    p_baton.x_range.range_padding = 0.1
    p_baton.xaxis.major_label_orientation = 1
    p_baton.xgrid.grid_line_color = None
        
    return p_baton    
    pass
###############################################################################
#       representation du diagramme en baton des DH en fonction des k_erreurs => debut
###############################################################################
    
###############################################################################
#               affichage diagramme en baton et 
#                       le runtime  
#               dans des fichiers html differents 
#               ====> debut
###############################################################################
def affichage_batons_runtime_k_1(df_kerrs, k_erreurs, rep_visu,
                             facteur_width, facteur_height, TOOLS):
    """
    affichage diagramme en baton et 
                       le runtime  
            dans des fichiers html differents 
    """
    # configuration figure
    output_file(rep_visu+"/"+"runtime_dashboard.html");
    
    p_tps_cal = None;
    p_tps_cal = plot_bokeh_tps_calcul_k_1(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS)
    show(p_tps_cal)
    
    # configuration figure
    output_file(rep_visu+"/"+"baton_dh_dashboard.html");
    p_baton_dh = plot_bokeh_baton_dh_k_1(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS)
    show(p_baton_dh)
    
###############################################################################
#               affichage diagramme en baton et 
#                       le runtime  
#               dans des fichiers html differents 
#               ====> fin
###############################################################################

###############################################################################
#               representation histogramme de dc, dh avec bokeh => debut
###############################################################################
def plot_bokeh_dc_dh_1(moy_dc_dh, df_grouped, k_erreur, TOOLS):
    """
    representer l'histogramme des distances dc et dh
    """
    label_dc_dh = ""
    if moy_dc_dh == "dc":
        label_dc_dh = "moy_dc";
    elif moy_dc_dh == "dh":
        label_dc_dh = "moy_dh";
    
    mu = df_grouped[moy_dc_dh].mean(); 
    sigma = df_grouped[moy_dc_dh].std();
    title, xlabel, ylabel = fct_aux_vis.title_xlabel_ylabel_figure(
                                label_dc_dh, k_erreur, mu, sigma, 
                                BOOL_ANGLAIS)
    
    print("moy_dc_dh max={},min={}".format(max(df_grouped[moy_dc_dh]),min(df_grouped[moy_dc_dh])))
    arr_hist, edges = None, None;
    if min(df_grouped[moy_dc_dh]) != max(df_grouped[moy_dc_dh]) :
        arr_hist, edges = np.histogram(df_grouped[moy_dc_dh],
                        bins=int( np.rint(max(df_grouped[moy_dc_dh]) / 1) ), 
                        range=[int( np.rint(min(df_grouped[moy_dc_dh])) ), 
                               int(max(df_grouped[moy_dc_dh]))]
                        )
    else:
        arr_hist, edges = np.histogram(df_grouped[moy_dc_dh],
                        bins=1, 
                        range=[int(np.rint(min(df_grouped[moy_dc_dh]))), 
                               int(max(df_grouped[moy_dc_dh])) + 1]
                        )
    edges = np.array( list(map(np.rint, edges)) )
    edges = np.array( list(map(int, edges)) )
    arr_hist = np.array( list(map(int, arr_hist)) )
    print("k_erreur={},label_dc_dh:{} arr_hist={} \n edges={} \n".format(
            k_erreur, label_dc_dh, arr_hist[:3], edges[:3]))
            
    df_dc_dh_k = pd.DataFrame({label_dc_dh:arr_hist, 
                                'left':edges[:-1],
                                'right':edges[1:]})
    ## convert to percent
    df_dc_dh_k[label_dc_dh] = (df_dc_dh_k[label_dc_dh]/\
                               df_dc_dh_k[label_dc_dh].sum()) * 100
    df_dc_dh_k[label_dc_dh] = df_dc_dh_k[label_dc_dh].astype(int)
    
    
    # Create the blank plot
    p_dc_dh = figure(plot_height = HEIGHT, 
                     plot_width = WIDTH, 
                     title = title,
                     x_axis_label = xlabel, 
                     y_axis_label = ylabel, 
                     x_axis_type = "linear",
                     tools = TOOLS)
    # Add a quad glyph
    src = ColumnDataSource(df_dc_dh_k)
    p_dc_dh.quad(source = src, bottom=0, top=label_dc_dh, 
              left='left', right='right',
              fill_color='lightblue', line_color='black')
   
    # plot red dashed line and  yellow dashed line
    dico_mu = {"x_red":[k_erreur] * (max(df_dc_dh_k[label_dc_dh])+1),
               "x_yellow":[mu] * (max(df_dc_dh_k[label_dc_dh])+1),
                "y":range(0, max(df_dc_dh_k[label_dc_dh])+1, 1)
               }
    df_mu = pd.DataFrame(dico_mu, columns = ['x_red','x_yellow','y'])
    src = ColumnDataSource(df_mu);
    p_dc_dh.line(source=src, x='x_red', y='y', 
                 line_color="red", line_dash="dashed", line_width=3)
    p_dc_dh.line(source=src, x='x_yellow', y='y', 
                 line_color="yellow", line_dash="dashed", line_width=3)
    
    # Add a hover tool referring to the formatted columns
    hover = None;
    if label_dc_dh == "moy_dc":
        hover = HoverTool(tooltips = [
                                    ('correction distance', '@moy_dc'),
                                    ('(min_dc, max_dc)', '($x, $y)') 
                                    ]
                            )
    elif label_dc_dh == "moy_dh":
        hover = HoverTool(tooltips = [
                                    ('correction distance', '@moy_dh'),
                                    ('(min_dh, max_dh)', '($x, $y)') 
                                    ]
                            )

    # Add the hover tool to the graph
    p_dc_dh.add_tools(hover)
    
    
    
    return p_dc_dh;
    
    pass
###############################################################################
#               representation histogramme de dc, dh avec bokeh => fin
###############################################################################

###############################################################################
#               distribution avec bokeh avec 
#                   representation runtime
#                   diagramme en baton des DH_k en fonction des k_erreurs 
#               ====> debut
###############################################################################
def distribution_bokeh_avec_runtime_baton_k_1(
                        critere_correction, 
                         mode_correction,
                         prob, 
                         k_erreurs,
                         nbre_sommets_GR,
                         rep_):
    """
    visualiser les distributions de moy_dc, moy_dh, correl_dc_dh, cumul_dh
    pour moy_DC = 1;
    """
    facteur_width, facteur_height = 1.0, 1.0;
    
    rep, rep_dist = "", "";
    df_kerrs = None;
    p = None;
    rep = rep_ + "/" \
          + mode_correction + "/" \
          + critere_correction + "_sommets_GR_" + str(nbre_sommets_GR) + "/" \
          + DATA_P_REP + str(prob);
    rep_dist = rep + "/" + "distribution" + "/";
    df_kerrs = fct_aux_vis.create_dataframe_data_revue(rep_dist, k_erreurs, 
                                               DISTRIB_ROOT_FILE,
                                               DISTRIB_EXT_FILE,
                                               NAMES_HEADERS);
#    df_kerrs_dc_1 = df_kerrs[(df_kerrs['dc'] == 1)];
    
#    return df_kerrs, p;
    
    # creation repertoire visualisation
    rep_visualisation = rep_dist+"../../visualisation";
    path = Path(rep_visualisation); path.mkdir(parents=True, exist_ok=True);
    
    # configuration figure
    output_file(rep_visualisation+"/"+"distribution_dashboard_dc_1.html");
    
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select";
        
    p_cols = []
    
    for ind, k_erreur in enumerate(k_erreurs):
        p_cols_k = [];
        df_kerrs_k = df_kerrs[df_kerrs["k_erreur"] == k_erreur]
        if k_erreur == 1:
            df_kerrs_k = df_kerrs_k[df_kerrs_k["dc"] == 1]
        df_grouped_numGraph = df_kerrs_k.groupby("num_graph")\
                                ["dc",'dh',"correl_dc_dh",'runtime'].mean()
                                
        # moy_dc
        p_dc = None;
        p_dc = plot_bokeh_dc_dh_1("dc", df_grouped_numGraph, k_erreur, TOOLS)
        p_cols_k.append(p_dc)
        
        #moy_dh
        p_dh = None;
        p_dh = plot_bokeh_dc_dh_1("dh", df_grouped_numGraph, k_erreur, TOOLS)
        p_cols_k.append(p_dh)
        
        # moy_dh cumul 
        p_cumul_dh = None;
        p_cumul_dh = bkh.plot_bokeh_cumul_corr_dc_dh(
                            "cumul_dh", df_grouped_numGraph, k_erreur, TOOLS)
        p_cols_k.append(p_cumul_dh)
        
        #correl cumul dc dh
        p_cumul_corr = None;
        p_cumul_corr = bkh.plot_bokeh_cumul_corr_dc_dh(
                        "cumul_correl", df_grouped_numGraph, k_erreur, TOOLS)
        p_cols_k.append(p_cumul_corr)
#    else:
#        pass
        
        p_cols.append(p_cols_k);
        
    # tps de calcul
    p_tps_cal = None;
    p_tps_cal = plot_bokeh_tps_calcul_k_1(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS)
    p_cols.append([p_tps_cal]);
    
    # diagramme en batons
    p_baton_dh = None;
    p_baton_dh = plot_bokeh_baton_dh_k_1(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS)
    p_cols.append([p_baton_dh]);

#    p = gridplot(p_cols, toolbar_location='above')
    p = gridplot(p_cols, toolbar_location=None)
    show(p)
    
    facteur_width, facteur_height = 3.0, 1.5;
    affichage_batons_runtime_k_1(df_kerrs, k_erreurs, rep_visualisation,
                             facteur_width, facteur_height, TOOLS)
                                     
    return df_kerrs, p;
    pass

if __name__ == '__main__':
    start = time.time();
    
    nbre_sommets_graphes = [15]; #[15] or [12];
    modes_correction = ["lineaire_simul50Graphes_priorite_aucune"];
    criteres_correction = ["aleatoire"];
    
    reps = ["/home/willy/Documents/python_topology_learning_simulation_debug/"
            +"correction_k_1/data"]
    probs = [0]
    k_erreurs = [1]
#    k_erreurs = [1,2,5,7,10,20,40]
    
    tuple_caracteristiques = [];
    for crit_mod_prob_k_nbreGraph_rep in it.product(
                                        criteres_correction,
                                        modes_correction,
                                        probs,
                                        [k_erreurs],
                                        nbre_sommets_graphes,
                                        reps):
        critere = crit_mod_prob_k_nbreGraph_rep[0];
        mode = crit_mod_prob_k_nbreGraph_rep[1];
        prob = crit_mod_prob_k_nbreGraph_rep[2];
        k_erreurs = crit_mod_prob_k_nbreGraph_rep[3];
        nbre_sommets_graphe = crit_mod_prob_k_nbreGraph_rep[4];
        
        tuple_caracteristiques.append(crit_mod_prob_k_nbreGraph_rep)
        
    
    BOOL_MATPLOTLIB_BOKEH_PLOT = True        # True : bokeh, False: matplotlib
    df_kerrs = None; 
    p = None;    
    for tuple_caracteristique in tuple_caracteristiques:
        if BOOL_MATPLOTLIB_BOKEH_PLOT:
            df_kerrs, p = distribution_bokeh_avec_runtime_baton_k_1(
                            *tuple_caracteristique);
#            df_kerrs, p = bkh.affichage_baton_runtime_meme_fichier_html(
#                            *tuple_caracteristique)
        else:
#            df_kerrs = matplib.distribution_matplotlib(*tuple_caracteristique)
            df_kerrs = matplib.distribution_matplotlib_with_chunksize_k(
                            *tuple_caracteristique)
#            
        
        pass
        
    print("runtime:{}".format(time.time() - start))
    
    
    