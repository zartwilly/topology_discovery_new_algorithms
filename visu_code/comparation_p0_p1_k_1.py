#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:39:43 2019

@author: willy
"""
import time;
import numpy as np;
import pandas as pd;

import fonctions_auxiliaires_visu as fct_aux_vis;

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
               "P", "*", "h", "H", "+", "x", "X", "D", "d"];

COLORS = ["red", "yellow", "blue", "green", "rosybrown", 
              "darkorange", "fuchsia", "grey"]
           
TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select";
BOOL_ANGLAIS = True;
###############################################################################
#                       CONSTANTES  ===> fin
###############################################################################

def create_dataframe_for_comparaison_p0_p05(dico_reps):
    """
    creer un dataframe contenant les DC, DH, CORREL_DC_DH, RUNTIME
    """
    df_probs = pd.DataFrame();
    list_dfs = []
    for prob, rep in dico_reps.items():
        rep_dist = rep + "/"+ "distribution/";
        k_erreurs = fct_aux_vis.lire_k_erreurs(rep_dist);
        df_prob = fct_aux_vis.create_dataframe_data_revue(
                                rep_dist, 
                                k_erreurs, 
                                DISTRIB_ROOT_FILE,
                                DISTRIB_EXT_FILE, 
                                NAMES_HEADERS);
        df_prob_grouped_numGraph = df_prob.groupby("num_graph")\
                                ["dc",'dh',"k_erreur","correl_dc_dh",'runtime'].mean()
        df_prob_grouped_numGraph.sort_values(['dc','dh','k_erreur'], 
                                              ascending=[True,True,True],
                                              inplace=True);
                                             
        df_prob_grouped_numGraph.reset_index(inplace=True);
        df_prob_grouped_numGraph['k_erreur'] = \
                            df_prob_grouped_numGraph['k_erreur'].astype(int);
                            
        headers_prob = dict();
        for col in df_prob_grouped_numGraph.columns:
            print("col={}".format(col));
            if col == "k_erreur":
                continue;
            headers_prob[col] = "_".join([col,str(prob)])
        df_prob_grouped_numGraph = df_prob_grouped_numGraph.rename(
                                        columns = headers_prob);
        list_dfs.append(df_prob_grouped_numGraph);
        
    # suppression col=k_erreur dans tous les dataframes sauf 1 
    f_shape = lambda df: df.shape[0];
    df_max_shape = max(map(f_shape, list_dfs));
    list_dfs_new = [];
    for id_df, df in enumerate(list_dfs):
        if df.shape[0] != df_max_shape:
            df.drop('k_erreur',axis=1, inplace=True);
        list_dfs_new.append(df);
        
    # merge df_prob in list_dfs
    df_probs = pd.concat(list_dfs, axis=1, sort=False, join='outer');
    
    return df_probs;

###############################################################################
#          plot comparaison de dc_0, dc_0.5 pour k_erreur ---> debut
###############################################################################
def plot_compa_cols_k(df_k, dcs, k_erreur):
    """
    """
    title = "comparaison "+ ", ".join(dcs) + " , k = "+ str(k_erreur)
    xlabel = "graph number";
    ylabel = "_".join(dcs);
    p_dcs = figure(plot_height = int(HEIGHT * 1.0), 
                    plot_width = int(WIDTH * 1.0), 
                    title = title,
                    x_axis_label = xlabel, 
                    y_axis_label = ylabel, 
                    tools = TOOLS);
    
    src = ColumnDataSource(df_k);

    for i, dc in enumerate(dcs):
        p_dcs.line(x="graph_number", y=dcs[i], source=src, 
                   color = COLORS[i],
                   legend= " "+str(dcs[i]),
                   line_width=2, line_dash="dashed");
    
    p_dcs.legend.location = "top_left";
    return p_dcs;
                   
###############################################################################
#          plot comparaison de dc_0, dc_0.5 pour k_erreur ---> debut
###############################################################################
    
###############################################################################
#               plot comparaison de dc_0, dc_0.5 ---> debut
###############################################################################
def plot_comparaison_bokeh(df_probs, rep_visualisation):
    """
    courbe pour comparer les dc_0, dc_0.5
                             dh_0, dh_0.5
                             correl_dc_dh_0, correl_dc_dh_0.5
    """
    for k_erreur in df_probs["k_erreur"].unique():
        df_k = df_probs[df_probs["k_erreur"] == k_erreur];
        df_k.fillna(-5, inplace=True);
        df_k['graph_number'] = np.arange(1, df_k.shape[0]+1);
        
        
        output_file(rep_visualisation+"/"
                    +"comparaison_dashboard_p0_p05_k_"+str(k_erreur)+".html");
        #dc
        dcs = ["dc_0","dc_0.5"];
        p_dc = plot_compa_cols_k(df_k, dcs, k_erreur);
        #dh
        dhs = ["dh_0","dh_0.5"];
        p_dh = plot_compa_cols_k(df_k, dhs, k_erreur);
        #correl
        correls = ["correl_dc_dh_0","correl_dc_dh_0.5"];
        p_correl = plot_compa_cols_k(df_k, correls, k_erreur);
        #runtime
        runtimes = ["runtime_0","runtime_0.5"];
        p_runtime = plot_compa_cols_k(df_k, runtimes, k_erreur);
        
        p = gridplot([[p_dc, p_dh, p_correl, p_runtime]], 
                     toolbar_location='above')
        show(p);
###############################################################################
#               plot comparaison de dc_0, dc_0.5 ---> fin
###############################################################################

if __name__ == '__main__':
    start = time.time();
    
    nbre_sommets_GRs = [15]; #[15] or [12];
    modes_correction = ["lineaire_simul50Graphes_priorite_aucune"];
    criteres_correction = ["aleatoire"];
    
    prob_0 = 0; prob_05 = 0.5
    dico_reps = {
        prob_0:"/home/willy/Documents/python_topology_learning_simulation_debug/"+
            "correction_k_1/data/lineaire_simul50Graphes_priorite_aucune/"+
            "aleatoire_sommets_GR_15/data_p_"+str(prob_0),
        prob_05:"/home/willy/Documents/python_topology_learning_simulation/"+
            "data_repeat/lineaire_simul50Graphes_priorite_aucune/"+
            "aleatoire_sommets_GR_15/data_p_"+str(prob_05)
            }
    
    rep_ = "/home/willy/Documents/python_topology_learning_simulation_debug/" \
            + "correction_k_1/data"
    rep = rep_ + "/" \
          + modes_correction[0] + "/" \
          + criteres_correction[0] + "_sommets_GR_" + str(nbre_sommets_GRs[0]);
    
    
    # creation repertoire visualisation
    rep_visualisation = rep+"/visualisation/comparaison_P0_P05";
    path = Path(rep_visualisation); path.mkdir(parents=True, exist_ok=True);
    
    df_probs = create_dataframe_for_comparaison_p0_p05(dico_reps);
    
    plot_comparaison_bokeh(df_probs, rep_visualisation);
    
    