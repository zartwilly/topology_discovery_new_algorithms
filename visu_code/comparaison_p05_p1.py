#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:41:13 2019

@author: willy

comparer DC, DH de k=[1,2,5,10,20] pour p = 0.5 et p = 1.0

"""

import os, time;
import numpy as np;
import pandas as pd;

import fonctions_auxiliaires_visu as fct_aux_viz;

from pathlib import Path;
#from scipy.stats import norm;
from matplotlib import lines, markers;

# bokeh
from bokeh.plotting import *
from bokeh.io import export_png;
from bokeh.layouts import grid, column;
from bokeh.models import ColumnDataSource, FactorRange;
from bokeh.plotting import figure, output_file, show, gridplot;
from bokeh.core.properties import value;
from bokeh.palettes import Spectral5;
from bokeh.models.tools import HoverTool;
from bokeh.models.tickers import FixedTicker;
from bokeh.models import FuncTickFormatter;

###############################################################################
#                           CONSTANTES --> debut
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
COLORS = ["red", "yellow", "blue", "green", "rosybrown", 
          "darkorange", "fuchsia", "grey"]

BOOL_ANGLAIS = True;
###############################################################################
#                           CONSTANTES --> fin
###############################################################################

###############################################################################
#                           lire k_erreurs --> debut
###############################################################################
def lire_k_erreurs(rep, p_s):
    """
    recherche k_erreurs qui sont en commun entre les p_x (x \in {0.5, 1.0}) 
    """
    k_s = set();
    for p in p_s:
        rep_dist = rep + "/" + DATA_P_REP + str(p) + "/" +"distribution" ;
        k = frozenset([int(file.split("_")[-1].split(".")[0]) 
            for file in os.listdir(rep_dist)]);
        k_s.add(k);
    k_erreurs = frozenset.intersection(*k_s);
    
    return set(k_erreurs);
###############################################################################
#                           lire k_erreurs --> fin
###############################################################################
    
###############################################################################
#                    creer un dataframe pour les p --> debut
###############################################################################
def create_dataframe(rep, p_s, k_erreurs):
    """
    creer un dataframe contenant les valeurs de chapue p_x (x \in {0.5, 1.0}) 
    """
    df = pd.DataFrame();
    
    frames = [];
    for p in p_s:
        rep_dist = rep + "/"+ DATA_P_REP + str(p) + "/" + "distribution" +"/"
        df_p = fct_aux_viz.create_dataframe_data_revue(
                rep_dist, k_erreurs,
                DISTRIB_ROOT_FILE,
                DISTRIB_EXT_FILE, 
                NAMES_HEADERS)
        df_p = df_p.groupby("num_graph").mean();
        df_p["p"] = p;
        frames.append(df_p);
        
    df = pd.concat(frames, axis = 0, ignore_index=True);
    dico_rename = {"dc":"moy_dc", "dh":"moy_dh"};
    df.rename(columns = dico_rename, inplace = True);
    
    return df;
###############################################################################
#                    creer un dataframe pour les p--> fin
###############################################################################
    
###############################################################################
#                           plot_bkh_comparaison_p --> debut
###############################################################################
def plot_bkh_comparaison_p(df, p_s, k_erreurs, var_cols, rep):
    """
    representer les comparaisons de p_x (x \in {0.5, 1.0}) selon les k_erreurs
    """
    
    title_root = "Comparaison DC, DH entre p = {}".format(p_s);
    xlabel = "graph number";
    ylabel = "";
    p_str = "_".join( map(lambda x:"".join(x.split(".")),map(str,p_s)) );
    
    # creation repertoire visualisation
    rep_visu = rep + "/" + "visualisation" + "/" + "comparaision_p_"+ p_str;
    path = Path(rep_visu); path.mkdir(parents=True, exist_ok=True);
    
    # configuration figure
    output_file(rep_visu+"/"+"comparaison_p_{}_dashboard.html".format(p_str));
    TOOLS = ""; #"pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select";
    
    facteur_width, facteur_height = 1.5, 1.5;
    frame_p = [];
    
    for k_erreur in k_erreurs:
        
        print("k = {} ---> debut".format(k_erreur))
        
        p_k = [];
        for var_dc_dh in var_cols:
            # configuration label title
            title = title_root + " " + "pour k = {}".format(k_erreur);
            ylabel = var_dc_dh;
        
            p_k_dc_dh = figure(plot_height = int(HEIGHT * facteur_height), 
                               plot_width = int(WIDTH * facteur_width), 
                               title = title,
                               x_axis_label = xlabel, 
                               y_axis_label = ylabel, 
                               tools = TOOLS);
            
            for ind_p, p in enumerate(p_s):
                df_k_p = df[ (df["k_erreur"] == k_erreur) & (df["p"] == p) ];
                df_k_p = df_k_p.sort_values(by=[var_dc_dh], axis = 0);
                df_k_p['graph_number'] = np.arange(1, df_k_p.shape[0]+1);
                
                src = ColumnDataSource(df_k_p);
                p_k_dc_dh.line(
                        x="graph_number", y=var_dc_dh, source=src, 
                        color=COLORS[ind_p],
                        muted_color=COLORS[ind_p], muted_alpha=0.15,
                        legend= "p = "+str(p),
                        line_width=2, line_dash="dashed");
                p_k_dc_dh.legend.click_policy = "mute";
                p_k_dc_dh.legend.location = "top_left";
#                p_k_dc_dh.legend.orientation="horizontal"
            p_k.append(p_k_dc_dh);
        
        frame_p.append(p_k);
        print("k = {} ---> fin".format(k_erreur))
        pass
    
    p = gridplot(frame_p, 
                 toolbar_location='above');
    
    show(p);
    export_png(p, filename=rep_visu+"/"+"comparaison_p_{}.png".format(p_str));
    pass
###############################################################################
#                           plot_bkh_comparaison_p --> fin
###############################################################################

if __name__ == "__main__":
    
    rep_base = "/home/willy/Documents/python_topology_learning_simulation/data_repeat";
    type_priorite = "lineaire_simul50Graphes_priorite_aucune"
    mode_correction = "aleatoire"
    nombre_graphe = 15
    
    start = time.time();
    rep = rep_base + "/" \
          + type_priorite + "/" \
          + mode_correction + "_sommets_GR_" + str(nombre_graphe)
    p_s = [0.5, 1.0];
    var_cols = ["moy_dc", "moy_dh"];
    
    k_erreurs = sorted(lire_k_erreurs(rep, p_s));
    print("k_erreurs = {}".format(k_erreurs));
    
    df = create_dataframe(rep, p_s, k_erreurs);
    
    bool_bkh = True;
    
    if bool_bkh:
        plot_bkh_comparaison_p(df, p_s, k_erreurs, var_cols, rep);
#    else:
#        plot_matplotlib_comparaison_p(df, p_s, k_erreurs, rep);
    
    print("runtime = {}".format(time.time() - start));


