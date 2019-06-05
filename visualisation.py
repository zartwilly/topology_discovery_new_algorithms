#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:12:20 2019

@author: willy
"""
import math;
import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;
import itertools as it;
import fonctions_auxiliaires as fct_aux;
import matplotlib.font_manager as font_manager;

from pathlib import Path;
from scipy.stats import norm;
from matplotlib import lines, markers;

from bokeh.plotting import *
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, gridplot;
from bokeh.core.properties import value
from bokeh.palettes import Spectral5
from bokeh.models.tools import HoverTool
from bokeh.models.tickers import FixedTicker
from bokeh.models import FuncTickFormatter

NBRE_ROWS = 3;
NBRE_COLS = 2;
NBRE_SOMMETS_1 = 2;

DISTRIB_ROOT_FILE = "distribution_moyDistLine_moyHamming_k_";
DISTRIB_EXT_FILE = ".txt";
RESUM_ROOT_FILE = "resumeExecution_";
REUM_EXT_FILE = ".csv";
DATA_P_REP = "data_p_";

NAMES_HEADERS_DISTRIB = ("G_k", "k_erreur","moy_dc","moy_dh","nbre_sommets_LG",\
                         "nbre_aretes_LG","correl_dc_dh")

###############################################################################
#               distribution avec seaborn => debut
###############################################################################
def create_dataframe(rep_dist, k_erreurs):
    """
    creer un dataframe contenant les caracteristiques des k_erreurs.
    """
    df = pd.DataFrame();
    f = lambda st: "_".join([st.split("_")[0], st.split("_")[1],
                             st.split("_")[3], st.split("_")[4]])
    for k_erreur in k_erreurs:
        names_headers = [name+"_"+str(k_erreur) 
                            for name in NAMES_HEADERS_DISTRIB if name != "G_k"];
        names_headers.insert(0,"G_k")
        df_k = pd.read_csv(
                    rep_dist+DISTRIB_ROOT_FILE+str(k_erreur)+DISTRIB_EXT_FILE,
                    names=names_headers,
                    sep=";"
                         );
        df_k["G_k"] = df_k['G_k'].apply(f);
        df_k["moy_dc_"+str(k_erreur)] = df_k["moy_dc_"+str(k_erreur)].astype(int);
        df_k["moy_dh_"+str(k_erreur)] = df_k["moy_dh_"+str(k_erreur)].astype(int);
        df_k["correl_dc_dh_"+str(k_erreur)] = df_k["correl_dc_dh_"+str(k_erreur)].astype(int);
        if df.empty:
            df = df_k;
        else:
            df = pd.merge(left=df,right=df_k,left_on='G_k',right_on="G_k")
    
    return df;
    pass

def create_dataframe_data_revue(rep_dist, k_erreurs):
    """
    creer le dataframe, issue des donnees pour la revue scientifique 
    contenant les caracteristiques des k_erreurs.
    """
    df = pd.DataFrame();
    NAMES_HEADERS = ["G_k", "k_erreur", "moy_dc", "moy_dh", 
                     "aretes_matE", "correl_dc_dh"]
    
    f = lambda row: "_".join([row["G_k"].split("_")[0], str(row["num_line"])])
    
    for k_erreur in k_erreurs:
        names_headers = [name+"_"+str(k_erreur) 
                            for name in NAMES_HEADERS if name != "G_k"];
        names_headers.insert(0,"G_k")
#        print("names_headers={}".format(names_headers))
        
        df_k = pd.read_csv(
                    rep_dist+DISTRIB_ROOT_FILE+str(k_erreur)+DISTRIB_EXT_FILE,
                    names=names_headers,
                    sep=";"
                         );
        df_k["num_line"] = df_k.index + 1;
        df_k["G_k"] = df_k[['G_k','num_line']].apply(f, axis=1);
        df_k["moy_dc_"+str(k_erreur)] = df_k["moy_dc_"+str(k_erreur)].astype(int);
        df_k["moy_dh_"+str(k_erreur)] = df_k["moy_dh_"+str(k_erreur)].astype(int);
        df_k["correl_dc_dh_"+str(k_erreur)] = df_k["correl_dc_dh_"+str(k_erreur)];
        if df.empty:
            df = df_k;
        else:
#            print("shape: df:{}, df_k:{}".format(df.shape, df_k.shape))
            df = pd.merge(left=df,right=df_k,left_on='G_k',right_on="G_k")
    
    print("df shape={} => end".format(df.shape))        
    return df
    pass

def distribution_seaborn(critere_correction, 
                         mode_correction,
                         prob, 
                         k_erreurs,
                         nbre_sommets_GR,
                         rep_, 
                         bool_data_revue):
    """
    visualiser les distributions de moy_dc, moy_dh, correl_dc_dh, cumul_dh.
    """
    rep, rep_dist = "", "";
    df_kerrs = None;
    if not bool_data_revue:
        rep = rep_ + "/" \
              + critere_correction +"_sommets_GR_" + str(nbre_sommets_GR) + "/" \
              + mode_correction + "/" \
              + DATA_P_REP + str(prob);
        rep_dist = rep + "/" + "distribution" + "/";
        df_kerrs = create_dataframe(rep_dist, k_erreurs); 
    else:
        rep = rep_ + "/" \
              + mode_correction + "/" \
              + critere_correction + "/" \
              + DATA_P_REP + str(prob);
        rep_dist = rep + "/" + "distribution" + "/";
        df_kerrs = create_dataframe_data_revue(rep_dist, k_erreurs);
    
    # configuration figure
    fig = plt.figure();
    fig, axarr = plt.subplots(len(k_erreurs), 4);
    fig.subplots_adjust(hspace=0.5, wspace=0.6)
    default_size = fig.get_size_inches()
    fig.set_size_inches( (default_size[0]*2.5, default_size[1]*4.5) )
    print("w =", default_size[0], " h = ",default_size[1])
    
    #plot 
    for ind, k_erreur in enumerate(k_erreurs):
        print("k_erreur = {} => treating...".format(k_erreur))
        min_, max_ = 0, np.inf;
        max_ = max(df_kerrs["moy_dc_"+str(k_erreur)]) \
                if max(df_kerrs["moy_dc_"+str(k_erreur)]) >= max(df_kerrs["moy_dh_"+str(k_erreur)]) \
                else max(df_kerrs["moy_dh_"+str(k_erreur)])
        nb_inter = max_/2+1 if max_%2 == 0 else math.ceil(max_/2);
        print("min_={}, max_={}, nb_inter={}".format(min_, max_, nb_inter))
        bins = np.around((np.linspace(min_, max_, nb_inter)), decimals = 1);
        bins = [int(bin_) for bin_ in bins]
        print("bins={} \n".format(bins))
        
        ## ind = 0, axarr = 0 --> moy_dc
        df_kerrs["moy_dc_"+str(k_erreur)].hist(bins = bins, ax = axarr[ind,0])
        mu = df_kerrs["moy_dc_"+str(k_erreur)].mean(); 
        sigma = df_kerrs["moy_dc_"+str(k_erreur)].std();
        axarr[ind,0].set(
                xlabel= "moy_distance_correction", \
                ylabel= "nombre_graphe", \
                title = "distance de correction pour \n" \
                    + str(k_erreur) \
                    +" arete(s) modifiee(s) \n $\mu=%.3f,\ \sigma=%.3f\ $ " \
                    %(mu, sigma)
                        );
        print("yticks={}, xticks={}, count={}".format(
                    axarr[ind,0].get_yticks(), 
                    axarr[ind,0].get_xticks(), 
                    df_kerrs["moy_dc_"+str(k_erreur)].count()) )
        axarr[ind,0].axvline(x=k_erreur, color = 'r');
        axarr[ind,0].set_xticks(ticks=bins, minor=True);
        axarr[ind,0].set_yticks(ticks=axarr[ind,0].get_yticks());
        axarr[ind,0].set_yticklabels(
                ['{:3.2f}%'.format(
                        x*100/df_kerrs["moy_dc_"+str(k_erreur)].count()) \
                    for x in axarr[ind,0].get_yticks()]
                                    );
        
        
        ## ind = 0, axarr = 1 --> moy_dh
        df_kerrs["moy_dh_"+str(k_erreur)].hist(bins = bins, ax = axarr[ind,1]);
        mu = df_kerrs["moy_dh_"+str(k_erreur)].mean(); 
        sigma = df_kerrs["moy_dh_"+str(k_erreur)].std();
        axarr[ind,1].set(
                xlabel= "moy_distance_hamming", \
                ylabel= "nombre_graphe", \
                title = "distance de Hamming pour \n"\
                    + str(k_erreur) \
                    + " arete(s) modifiee(s) \n $\mu=%.3f,\ \sigma=%.3f\ $." \
                    %(mu, sigma)
                        );
        axarr[ind,1].axvline(x=k_erreur, color = 'r')
        axarr[ind,1].set_xticks(ticks=bins, minor=True);                       # axarr[ind,1].set_xticklabels(bins, rotation=0)
        axarr[ind,1].set_yticks(ticks=axarr[ind,1].get_yticks());
        axarr[ind,1].set_yticklabels(
                ['{:3.2f}%'.format(
                        x*100/df_kerrs["moy_dh_"+str(k_erreur)].count()) \
                    for x in axarr[ind,1].get_yticks()]
                                    );
        
        # ind=0, axarr = 2 --> correl_dl_dh
        data_sort = df_kerrs["correl_dc_dh_"+str(k_erreur)]\
                    .sort_values(ascending = True);
        axarr[ind,2].step(data_sort, data_sort.cumsum())
        axarr[ind,2].set(
                xlabel= "correlation_DC_DH", 
                ylabel= "cumulative correlation", \
                title = "fonction de repartition de \n"\
                        +"correlation entre moy_dl et moy_dh \n pour "\
                        +str(k_erreur)+" cases modifiees."
                        );
        axarr[ind,2].set_yticklabels(
                ['{:3.2f}%'.format(
                        x*100/df_kerrs["correl_dc_dh_"+str(k_erreur)].count()) \
                    for x in axarr[ind,2].get_yticks()]
                                    );
        
        # ind=0, axarr = 3 --> cumul_dh
        df_kerrs.sort_values(by = "moy_dh_"+str(k_erreur), 
                             ascending=True, 
                             axis = 0, 
                             inplace = True);
        df_kerrs["nb_graphe_dh<x"] = \
            df_kerrs["moy_dh_"+str(k_erreur)]\
            .apply( lambda x: \
                   df_kerrs["moy_dh_"\
                            +str(k_erreur)][df_kerrs["moy_dh_"\
                                            +str(k_erreur)] < x].count()/\
                   df_kerrs["moy_dh_"\
                            +str(k_erreur)].count())
#        print("--->k={}, cumul_dh => min = {}, max = {},".format(k_error, df["nb_graphe_dh<x"].min(), df["nb_graphe_dh<x"].max()))
        axarr[ind,3].set(
                xlabel= "moy_DH", 
                ylabel= "number graph moy_DH < x ", \
                title = "cumulative moy_dh pour \n"+str(k_erreur)+" cases modifiees");
        axarr[ind,3].step(df_kerrs["moy_dh_"+str(k_erreur)],
                          df_kerrs["nb_graphe_dh<x"]);
        axarr[ind,3].set_xticklabels(np.arange(0, 
                                               df_kerrs["moy_dh_"+str(k_erreur)].count(), 
                                               10), 
                                     rotation=45);   
        
    # save axarr
    fig.tight_layout();
#    plt.grid(True);
    save_vizu = rep + "/../" + "visualisation";
    path = Path(save_vizu); path.mkdir(parents=True, exist_ok=True);
   
    fig.savefig(save_vizu+"/"+"distanceMoyenDLDH_k_"+"_".join(map(str,k_erreurs)) \
                +"_p_"+str(prob)+".jpeg", dpi=190);
                
    return df_kerrs;
    pass
###############################################################################
#               distribution avec seaborn => fin
###############################################################################

###############################################################################
#               distribution avec bokeh => debut
###############################################################################
WIDTH = 400;
HEIGHT = 400;
def distribution_bokeh(critere_correction, 
                         mode_correction,
                         prob, 
                         k_erreurs,
                         nbre_sommets_GR,
                         rep_, 
                         bool_data_revue):
    """
    visualiser les distributions de moy_dc, moy_dh, correl_dc_dh, cumul_dh.
    """
    rep, rep_dist = "", "";
    df_kerrs = None;
    if not bool_data_revue:
        rep = rep_ + "/" \
              + critere_correction +"_sommets_GR_" + str(nbre_sommets_GR) + "/" \
              + mode_correction + "/" \
              + DATA_P_REP + str(prob);
        rep_dist = rep + "/" + "distribution" + "/";
        df_kerrs = create_dataframe(rep_dist, k_erreurs); 
    else:
        rep = rep_ + "/" \
              + mode_correction + "/" \
              + critere_correction + "/" \
              + DATA_P_REP + str(prob);
        rep_dist = rep + "/" + "distribution" + "/";
        df_kerrs = create_dataframe_data_revue(rep_dist, k_erreurs);
    
    # configuration figure
    output_file(rep_dist+"../../visualisation/"+"distribution_dashboard.html");
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select";
        
    p_cols = []
    
    for ind, k_erreur in enumerate(k_erreurs):
        p_cols_k = [];
        
        ###### moy_dc
        mu = df_kerrs["moy_dc_"+str(k_erreur)].mean(); 
        sigma = df_kerrs["moy_dc_"+str(k_erreur)].std();
        title = "distance de correction pour \n" \
                + str(k_erreur) \
                +" arete(s) modifiee(s) \n $\mu=%.3f,\ \sigma=%.3f\ $ " \
                %(mu, sigma)
               
        arr_hist, edges = np.histogram(df_kerrs['moy_dc_'+str(k_erreur)],
                            bins=int(max(df_kerrs['moy_dc_'+str(k_erreur)]) / 2), 
                            range=[min(df_kerrs['moy_dc_'+str(k_erreur)]), 
                                   max(df_kerrs['moy_dc_'+str(k_erreur)])]
                            )
                
        df_dc_k = pd.DataFrame({"moy_dc":arr_hist, 
                                'left':edges[:-1],
                                'right':edges[1:]})
        # Create the blank plot
        p_dc = figure(plot_height = HEIGHT, plot_width = WIDTH, 
                      title = title,
                      x_axis_label = 'moy_dc', 
                      y_axis_label = 'nombre_graphe', 
                      tools = TOOLS
                    )
        # Add a quad glyph
        src = ColumnDataSource(df_dc_k)
        p_dc.quad(source = src, bottom=0, top='moy_dc', 
                  left='left', right='right',
                  fill_color='red', line_color='black')
        # Add a hover tool referring to the formatted columns
        hover = HoverTool(tooltips = [('correction distance', '@moy_dc'),
                                      ('(min_dc, max_dc)', '($x, $y)') ])
        # Style the plot
#        p_dc = style(p_dc)
        # TODO comment remplacer le nombre par un pourcentage
#        tickers = np.around(arr_hist * 100 / df_kerrs['moy_dc_'+str(k_erreur))
#        p_dc.yaxis.ticker = ['{:3.2f}%'.format(
#                        x*100/df_kerrs["moy_dc_"+str(k_erreur)].count()) \
#                    for x in p_dc.yaxis.get_yticks()]
        
#        p_dc.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
#        def ticker():
#            return '{:3.2f}%'.format(tick * 100/df_kerrs["moy_dc_"+str(k_erreur)].count())
#        p_dc.yaxis.formatter = FuncTickFormatter(
#                                code=""" 
#                                    return tick * 100/df_kerrs["moy_dc_"+str(k_erreur)].count()
#                                """)
                                    

        # Add the hover tool to the graph
        p_dc.add_tools(hover)
        
        p_cols_k.append(p_dc)
        
        ###### moy_dh
        mu = df_kerrs["moy_dh_"+str(k_erreur)].mean(); 
        sigma = df_kerrs["moy_dh_"+str(k_erreur)].std();
        title = "distance de Hamming pour \n" \
                + str(k_erreur) \
                +" arete(s) modifiee(s) \n $\mu=%.3f,\ \sigma=%.3f\ $ " \
                %(mu, sigma)
               
        arr_hist, edges = np.histogram(df_kerrs['moy_dh_'+str(k_erreur)],
                            bins=int(max(df_kerrs['moy_dh_'+str(k_erreur)]) / 2), 
                            range=[min(df_kerrs['moy_dh_'+str(k_erreur)]), 
                                   max(df_kerrs['moy_dh_'+str(k_erreur)])]
                            )
                
        df_dh_k = pd.DataFrame({"moy_dh":arr_hist, 
                                'left':edges[:-1],
                                'right':edges[1:]})
        # Create the blank plot
        p_dh = figure(plot_height = HEIGHT, plot_width = WIDTH, 
                      title = title,
                      x_axis_label = 'moy_dh', 
                      y_axis_label = 'nombre_graphe', 
                      tools = TOOLS)
        # Add a quad glyph
        src = ColumnDataSource(df_dh_k)
        p_dh.quad(source = src, bottom=0, top='moy_dh', 
                  left='left', right='right',
                  fill_color='red', line_color='black')
        # Add a hover tool referring to the formatted columns
        hover = HoverTool(tooltips = [('correction distance', '@moy_dh'),
                                      ('(min_dc, max_dh)', '($x, $y)') ])
        # Style the plot
#        p_dh = style(p_dh)

        # Add the hover tool to the graph
        p_dh.add_tools(hover)
        
        p_cols_k.append(p_dh)
        
        #### correl_dc_dh
        df_corr_k = df_kerrs[['moy_dh_'+str(k_erreur), 'moy_dc_'+str(k_erreur),
                              'correl_dc_dh_'+str(k_erreur)]];
        df_corr_k_sorted = df_corr_k.sort_values(
                                by='correl_dc_dh_'+str(k_erreur),
                                axis=0, ascending=True);
        df_corr_k_sorted['cumsum_k'] = df_corr_k_sorted['correl_dc_dh_'+str(k_erreur)].cumsum(axis=0);
        
        src = ColumnDataSource(df_corr_k_sorted)
        title = "fonction de repartition de \n" \
                +"correlation entre moy_dc et moy_dh \n pour " \
                +str(k_erreur)+" cases modifiees."
        p_corr_k = figure(plot_height = HEIGHT, plot_width = WIDTH, 
                      title = title,
                      x_axis_label = 'correlation_DC_DH', 
                      y_axis_label = 'cumulative correlation', 
                      tools = TOOLS)
        p_corr_k.line(source=src, 
                      x='correl_dc_dh_'+str(k_erreur),
                      y='cumsum_k')
        
        p_cols_k.append(p_corr_k)
        
        #### cumul_dh
        df_corr_k_sorted = df_corr_k.sort_values(
                                by='moy_dh_'+str(k_erreur),
                                axis=0, ascending=True);
        df_corr_k_sorted["nb_graphe_dh<x"] = \
            df_corr_k_sorted["moy_dh_"+str(k_erreur)]\
            .apply( lambda x: \
                   df_corr_k_sorted["moy_dh_"\
                            +str(k_erreur)][df_corr_k_sorted["moy_dh_"\
                                            +str(k_erreur)] < x].count()/\
                   df_corr_k_sorted["moy_dh_"\
                            +str(k_erreur)].count())
        
        src = ColumnDataSource(df_corr_k_sorted)
        title = "cumulative moy_dh pour \n"+str(k_erreur)+" cases modifiees";
        p_corr_sup_x_k = figure(plot_height = HEIGHT, plot_width = WIDTH, 
                      title = title,
                      x_axis_label = 'number graph moy_DH < x', 
                      y_axis_label = "cumulative moy_dh pour \n"+str(k_erreur)+" cases modifiees", 
                      tools = TOOLS)
        p_corr_sup_x_k.line(source=src, 
                      x="moy_dh_"+str(k_erreur),
                      y="nb_graphe_dh<x")
        
        p_cols_k.append(p_corr_sup_x_k)
                
        p_cols.append(p_cols_k);
        
    p = gridplot(p_cols)
    show(p)
    return df_kerrs
    
###############################################################################
#               distribution avec bokeh => fin
###############################################################################

if __name__ == '__main__':
    reps = ["tests"];
    nbre_sommets_graphes = [6]
    probs = [0.0];
    k_erreurs = [1,2,3,4];
    modes_correction = ["aleatoire_sans_remise"];
    criteres_correction = ["nombre_aretes_corrigees"];
    
    bool_data_revue = True;
    if bool_data_revue:
        reps = ["/home/willy/Documents/python_topology_learning_simulation/data_repeat_old"]
        probs = [0.5]
        modes_correction = ["lineaire_simul50Graphes_priorite_aucune"]
        criteres_correction = ["aleatoire"]
        k_erreurs = [1, 2, 5, 10, 15, 20] 
    else:
        reps = ["tests"];
    
    tuple_caracteristiques = [];
    for crit_mod_prob_k_nbreGraph_rep in it.product(
                                        criteres_correction,
                                        modes_correction,
                                        probs,
                                        [k_erreurs],
                                        nbre_sommets_graphes,
                                        reps,
                                        [bool_data_revue]):
        critere = crit_mod_prob_k_nbreGraph_rep[0];
        mode = crit_mod_prob_k_nbreGraph_rep[1];
        prob = crit_mod_prob_k_nbreGraph_rep[2];
        k_erreurs = crit_mod_prob_k_nbreGraph_rep[3];
        nbre_sommets_graphe = crit_mod_prob_k_nbreGraph_rep[4];
        
        tuple_caracteristiques.append(crit_mod_prob_k_nbreGraph_rep)
    
    df_kerrs=None    
    for tuple_caracteristique in tuple_caracteristiques:
#        df_kerrs = distribution_seaborn(*tuple_caracteristique);
        
        df_kerrs = distribution_bokeh(*tuple_caracteristique);
        
        