#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:36:56 2019

@author: willy
"""
import math;
import time;
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
#               representation histogramme de dc, dh avec bokeh => debut
###############################################################################
def plot_bokeh_dc_dh(moy_dc_dh, df_grouped, k_erreur, TOOLS):
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
    if min(df_grouped[moy_dc_dh]) != 0 and max(df_grouped[moy_dc_dh]) != 0:
        arr_hist, edges = np.histogram(df_grouped[moy_dc_dh],
                        bins=int( np.rint(max(df_grouped[moy_dc_dh]) / 1) ), 
                        range=[int( np.rint(min(df_grouped[moy_dc_dh])) ), 
                               int(max(df_grouped[moy_dc_dh]))]
                        )
    else:
        arr_hist, edges = np.histogram(df_grouped[moy_dc_dh],
                        bins=1, 
                        range=[0,2]
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
    # Style the plot
#        p_dc_dh = style(p_dc_dh)

    # Add the hover tool to the graph
    p_dc_dh.add_tools(hover)
    
    
    
    return p_dc_dh;
    
    pass
###############################################################################
#               representation histogramme de dc, dh avec bokeh => fin
###############################################################################

###############################################################################
#               representation fonction cumulative avec bokeh => debut
###############################################################################
def plot_bokeh_cumul_corr_dc_dh(correl_dc_dh_cumul, 
                                df_grouped_numGraph, 
                                k_erreur, 
                                TOOLS):
    """
    representer la fonction de repartition cumulee de 
        la correlation entre dc et dh -> correl_dc_dh
        la distance de hamming -> dh
    """
    label_cumul, new_col_label = "", ""
    if correl_dc_dh_cumul == "cumul_correl":
        label_cumul = "correl_dc_dh"
        new_col_label = "nb_graphe_correl"
    elif correl_dc_dh_cumul == "cumul_dh":
        label_cumul = "dh"
        new_col_label = "nb_graphe_dh"
        
    df_corr_dc_sorted = df_grouped_numGraph.sort_values(
                        by=label_cumul, axis=0, ascending=True)
    
    mu = df_grouped_numGraph['dh'].mean(); 
#    sigma = df_grouped_numGraph['dh'].std();
    
    df_corr_dc_sorted[new_col_label+"<x"] = \
            df_corr_dc_sorted[label_cumul].apply( lambda x: \
                             df_corr_dc_sorted[label_cumul][df_corr_dc_sorted[label_cumul]<x].count()/\
                             df_corr_dc_sorted[label_cumul].count()
                             )
    
    src = ColumnDataSource(df_corr_dc_sorted);
    title, xlabel, ylabel = fct_aux_vis.title_xlabel_ylabel_figure(
                                correl_dc_dh_cumul, k_erreur, 0, 0,
                                BOOL_ANGLAIS);
    print("correl_dc_dh_cumul={}, title={}, xlabel={}".format(correl_dc_dh_cumul, title, xlabel))
                                                       
    p_corr_dc_dh_sup_x_k = figure(
                            plot_height = HEIGHT, 
                            plot_width = WIDTH, 
                            title = title,
                            x_axis_label = xlabel, 
                            y_axis_label = ylabel, 
                            tools = TOOLS)
    p_corr_dc_dh_sup_x_k.line(source=src, 
                  x=label_cumul,
                  y=new_col_label+"<x")
    
    #add vertical line at k_erreur for mu value
    if correl_dc_dh_cumul == "cumul_dh":
        dico_mu = {"x_red":[k_erreur] * 2,
               "x_yellow":[mu] * 2,
                "y":range(0, 2, 1)
               }
        df_mu = pd.DataFrame(dico_mu, columns = ['x_red','x_yellow','y'])
        src = ColumnDataSource(df_mu);
        p_corr_dc_dh_sup_x_k.line(source=src, x='x_red', y='y', 
                 line_color="red", line_dash="dashed", line_width=3)
        p_corr_dc_dh_sup_x_k.line(source=src, x='x_yellow', y='y', 
                 line_color="yellow", line_dash="dashed", line_width=3)
    elif correl_dc_dh_cumul == "cumul_correl":
        mu_corr = df_corr_dc_sorted[label_cumul].mean();
        dico_mu = {"x_yellow":[mu_corr] * 2,
                   "y":range(0, 2, 1)
                   }
        df_mu = pd.DataFrame(dico_mu, columns = ['x_yellow','y'])
        src = ColumnDataSource(df_mu);
        p_corr_dc_dh_sup_x_k.line(source=src, x='x_yellow', y='y', 
                 line_color="yellow", line_dash="dashed", line_width=3)
    
    return p_corr_dc_dh_sup_x_k                                           
    
        
###############################################################################
#               representation fonction cumulative avec bokeh => fin
###############################################################################
    
###############################################################################
#           representation du tps d'execution des k_erreurs => debut
###############################################################################
def plot_bokeh_tps_calcul(df_kerrs, k_erreurs, facteur_width, 
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
def plot_bokeh_baton_dh(df_kerrs, k_erreurs, facteur_width, 
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

def plot_bokeh_baton_moy_dh(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS):
    """
    representation du diagramme en baton des DH en fonction des k_erreurs.
    """
    
    df_grouped_numgraph = df_kerrs.groupby("num_graph")\
                                ["dc",'dh','k_erreur','correl_dc_dh','runtime'].mean()
    
    df_moy_dh = pd.DataFrame(columns = ['k_error','moy_dh'])
    for k_erreur in k_erreurs:
        df_grouped_k = df_grouped_numgraph[ 
                            df_grouped_numgraph["k_erreur"] == k_erreur ]
        df_moy_dh.loc[len(df_moy_dh)] = [k_erreur, df_grouped_k['dh'].mean()]
        
    print("df_moy_dh = \n{}".format(df_moy_dh))
    df_moy_dh['k_error'] = df_moy_dh['k_error'].astype(int);
    df_moy_dh['moy_dh'] = df_moy_dh['moy_dh'] / df_moy_dh['k_error'];
    
    title = "ratio DH/k" + " (b)" #"Average Hamming distance DH" + " (b)"
    xlabel = "k"
    ylabel = "DH_k"
    p_baton = figure(plot_height = int(HEIGHT*facteur_height), 
                     plot_width = int(WIDTH*facteur_width),
#                     x_range=FactorRange(*tuples_x),
                     title = title,
                     x_axis_label = xlabel, 
                     y_axis_label = ylabel, 
                     tools = TOOLS);
    
    src = ColumnDataSource(df_moy_dh);
    ks = [0]+list(df_moy_dh['k_error'])
#    p_baton.line( x=ks, y=ks, color = "yellow",
#                 legend= "y=k", line_width=2, 
#                 line_dash="dashed");
    p_baton.vbar(source=src, x='k_error', top='moy_dh', 
                 width=0.1, legend= "moy_DH");
        
    p_baton.legend.location = "top_left"
    return p_baton;

def plot_bokeh_baton_div_by_kerreur(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS):
    """
    representation du diagramme en baton des DH en fonction des k_erreurs.
    chaque DH est divise par k
    """
    df_grouped_numgraph = df_kerrs.groupby("num_graph")\
                            ["dc",'dh','k_erreur','correl_dc_dh','runtime'].mean()
    
    df_grouped = pd.DataFrame()
    frames = []
    for k_erreur in k_erreurs:
        df_grouped_k = df_grouped_numgraph[ 
                            df_grouped_numgraph["k_erreur"] == k_erreur ]
        df_grouped_k['dh'] = df_grouped_k.loc[:,'dh'] / \
                                df_grouped_k.loc[:,'k_erreur']
        
        df_gr_k_sorted = df_grouped_k.sort_values(by=['k_erreur','dh'], axis=0);
        df_gr_k_sorted['numero_graphe'] = np.arange(1, df_gr_k_sorted.shape[0]+1)
        frames.append(df_gr_k_sorted)
        
    df_grouped = pd.concat(frames)
    df_grouped["k_erreur"] = df_grouped["k_erreur"].astype(int)
    df_grouped["dh"] = df_grouped["dh"].astype(int)
    
    subset = df_grouped[["k_erreur","numero_graphe"]]
    tuples_x = [tuple((str(x[0]), str(x[1]))) for x in subset.values]
#    subset = df_grouped[["k_erreur"]]
#    tuples_x = [( str(x[0]) ) for x in subset.values]
    dh = list(df_grouped["dh"].values);
    
    src = ColumnDataSource(data=dict(x=tuples_x, dh=dh))
    
    title = "Hamming distance by k added/deleted edges" + " (b)"
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
#       representation du diagramme en baton des DH en fonction des k_erreurs => fin
###############################################################################

###############################################################################
#               affichage diagramme en baton et 
#                       le runtime  
#               dans des fichiers html differents 
#               ====> debut
###############################################################################
def affichage_batons_runtime(df_kerrs, k_erreurs, rep_visu,
                             facteur_width, facteur_height, TOOLS):
    """
    affichage diagramme en baton et 
                       le runtime  
            dans des fichiers html differents 
    """
    # configuration figure
    output_file(rep_visu+"/"+"runtime_dashboard.html");
    
    p_tps_cal = None;
    p_tps_cal = plot_bokeh_tps_calcul(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS)
    show(p_tps_cal)
    
    # configuration figure
    output_file(rep_visu+"/"+"baton_dh_dashboard.html");
    p_baton_dh = plot_bokeh_baton_dh(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS)
    show(p_baton_dh)
    
###############################################################################
#               affichage diagramme en baton et 
#                       le runtime  
#               dans des fichiers html differents 
#               ====> fin
###############################################################################

###############################################################################
#               affichage diagramme en baton et 
#                       le runtime  
#               dans le meme fichier html 
#               ====> debut
###############################################################################
def affichage_baton_runtime_meme_fichier_html(critere_correction, 
                         mode_correction,
                         prob, 
                         k_erreurs,
                         nbre_sommets_GR,
                         rep_):
    """
    visualiser le diagramme en baton et la courbe du runtime 
    dans le meme fichier html.
    """
    facteur_width, facteur_height = 1.0, 1.0;
    
    rep, rep_dist = "", "";
    df_kerrs = None;
    rep = rep_ + "/" \
          + mode_correction + "/" \
          + critere_correction + "_sommets_GR_" + str(nbre_sommets_GR) + "/" \
          + DATA_P_REP + str(prob);
    rep_dist = rep + "/" + "distribution" + "/";
    df_kerrs = fct_aux_vis.create_dataframe_data_revue(rep_dist, k_erreurs, 
                                               DISTRIB_ROOT_FILE,
                                               DISTRIB_EXT_FILE,
                                               NAMES_HEADERS);
    
    # creation repertoire visualisation
    rep_visualisation = rep_dist+"../../visualisation";
    path = Path(rep_visualisation); path.mkdir(parents=True, exist_ok=True);
    
    # configuration figure
    output_file(rep_visualisation+"/"+"runtime_barplot_dashboard.html");
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select";
        
    p_col_baton, p_col_runtime = None, None;
    
    p_col_runtime = plot_bokeh_tps_calcul(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS);
    p_col_baton = plot_bokeh_baton_moy_dh(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS);
    p_col_baton_div_k = None;
    p_col_baton_div_k = plot_bokeh_baton_div_by_kerreur(
                            df_kerrs, 
                            k_erreurs, 
                            facteur_width, 
                            facteur_height, 
                            TOOLS)

    p = gridplot([[p_col_runtime, p_col_baton]], 
                 toolbar_location='above')                                  
#    p = gridplot([[p_col_runtime, p_col_baton, p_col_baton_div_k]], 
#                 toolbar_location='above')
    show(p)
    
    from bokeh.io import export_png
    export_png(p, filename=rep_visualisation+"/"+"runtime_barplot.png")

    return df_kerrs, p;
    pass
###############################################################################
#               affichage diagramme en baton et 
#                       le runtime  
#               dans le meme fichier html 
#               ====> fin
###############################################################################

###############################################################################
#               distribution avec bokeh avec 
#                   representation runtime
#                   diagramme en baton des DH_k en fonction des k_erreurs 
#               ====> debut
###############################################################################
def distribution_bokeh_avec_runtime_baton(critere_correction, 
                         mode_correction,
                         prob, 
                         k_erreurs,
                         nbre_sommets_GR,
                         rep_):
    """
    visualiser les distributions de moy_dc, moy_dh, correl_dc_dh, cumul_dh.
    """
    facteur_width, facteur_height = 1.0, 1.0;
    
    rep, rep_dist = "", "";
    df_kerrs = None;
    rep = rep_ + "/" \
          + mode_correction + "/" \
          + critere_correction + "_sommets_GR_" + str(nbre_sommets_GR) + "/" \
          + DATA_P_REP + str(prob);
    rep_dist = rep + "/" + "distribution" + "/";
    df_kerrs = fct_aux_vis.create_dataframe_data_revue(rep_dist, k_erreurs, 
                                               DISTRIB_ROOT_FILE,
                                               DISTRIB_EXT_FILE,
                                               NAMES_HEADERS);
    
    # creation repertoire visualisation
    rep_visualisation = rep_dist+"../../visualisation";
    path = Path(rep_visualisation); path.mkdir(parents=True, exist_ok=True);
    
    # configuration figure
    output_file(rep_visualisation+"/"+"distribution_dashboard.html");
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select";
        
    p_cols = []
    
    for ind, k_erreur in enumerate(k_erreurs):
        p_cols_k = [];
        
        df_kerrs_k = df_kerrs[df_kerrs["k_erreur"] == k_erreur]
        df_grouped_numGraph = df_kerrs_k.groupby("num_graph")\
                                ["dc",'dh',"correl_dc_dh",'runtime'].mean()
                                
        # moy_dc
        p_dc = None;
        p_dc = plot_bokeh_dc_dh("dc", df_grouped_numGraph, k_erreur, TOOLS)
        p_cols_k.append(p_dc)
        
        #moy_dh
        p_dh = None;
        p_dh = plot_bokeh_dc_dh("dh", df_grouped_numGraph, k_erreur, TOOLS)
        p_cols_k.append(p_dh)
        
        # moy_dh cumul 
        p_cumul_dh = None;
        p_cumul_dh = plot_bokeh_cumul_corr_dc_dh(
                            "cumul_dh", df_grouped_numGraph, k_erreur, TOOLS)
        p_cols_k.append(p_cumul_dh)
        
        #correl cumul dc dh
        p_cumul_corr = None;
        p_cumul_corr = plot_bokeh_cumul_corr_dc_dh(
                            "cumul_correl", df_grouped_numGraph, k_erreur, TOOLS)
        p_cols_k.append(p_cumul_corr)
        
        
        p_cols.append(p_cols_k);
        
    # tps de calcul
    p_tps_cal = None;
    p_tps_cal = plot_bokeh_tps_calcul(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS)
    p_cols.append([p_tps_cal]);
    
    # diagramme en batons
    p_baton_dh = None;
    p_baton_dh = plot_bokeh_baton_dh(df_kerrs, k_erreurs, facteur_width, 
                        facteur_height, TOOLS)
    p_cols.append([p_baton_dh]);
#    p = gridplot(p_cols)
    
#    l = layout([
#  [bollinger],
#  [sliders, plot],
#  [p1, p2, p3],
#], sizing_mode='stretch_both')
    
#    p = gridplot(p_cols, sizing_mode='scale_width',toolbar_location='above',
#                 plot_width=600, plot_height=600)
    p = gridplot(p_cols, toolbar_location='above')
    show(p)
    
    facteur_width, facteur_height = 3.0, 1.5;
    affichage_batons_runtime(df_kerrs, k_erreurs, rep_visualisation,
                             facteur_width, facteur_height, TOOLS)
    return df_kerrs, p;
###############################################################################
#               distribution avec bokeh avec 
#                   representation runtime
#                   diagramme en baton des DH_k en fonction des k_erreurs 
#               ====> fin
###############################################################################
    