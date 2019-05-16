#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:21:58 2019

@author: willy
"""
###############################################################################
#                   Classe noeud
###############################################################################
class Noeud :
    """ Classe definissant un noeud du graphe.
    """
    noeuds_crees = 0;
    def __init__(self,
                 nom, 
                 etat = 0,
                 cliques_S_1 = 0,
                 voisins = frozenset(), 
                 ext_init = "", 
                 ext_final = ""):
        """ constructeur d'un noeud.
        """
        Noeud.noeuds_crees += 1;
        self.id = Noeud.noeuds_crees;
        self._nom = nom;
        self._etat = etat;
        self._cliques_S_1 = cliques_S_1;
        self._voisins = voisins;
        self.ext_init = ext_init;
        self.ext_final = ext_final;
        
    def __repr__(self):
        """Quand on entre notre objet dans l'interpréteur"""
#        return "Noeud: nom({}), etat({}), voisins({}),ext_init({}),ext_final({})".format(
#                self.nom, self.etat, self.voisins, self.ext_init, self.ext_final);
                
        return "Noeud: nom({})".format(self.nom) +\
                " etat({})".format(self.etat) +\
                " cliques_S_1({})".format(self.cliques_S_1) +\
                " voisins({})".format(self.voisins) +\
                " ext_init({})".format(self.ext_init) +\
                " ext_final({})".format(self.ext_final) ;
                
    def combien_noeud(cls) :    
        """Méthode de classe affichant combien d'objets ont été créés"""
        print("Jusqu'à présent, {} noeuds ont été créés.".format(
                cls.noeuds_crees))
    combien_noeud = classmethod(combien_noeud);
    
    def _get_nom(self) :
        """ Methode affichant le nom du noeud. 
        """
        return self._nom;
    
    def _set_nom(self, nouveau_nom) :
        """ Methode pour modifier le nom du sommet.
        """
        self._nom = nouveau_nom;
    
    nom = property(_get_nom, _set_nom);
    
    def _get_voisins(self) :
        """ Methode affichant le nom du noeud. 
        """
        return self._voisins;
    
    def _set_voisins(self, nouveaux_voisins) :
        """ Methode pour modifier le nom du sommet.
        """
        self._voisins = nouveaux_voisins;
    
    voisins = property(_get_voisins, _set_voisins);
    
    def _get_etat(self) :
        """ Methode affichant le nom du noeud. 
        """
        return self._etat;
    
    def _set_etat(self, nouveau_etat) :
        """ Methode pour modifier le nom du sommet.
        """
        self._etat = nouveau_etat;
    
    etat = property(_get_etat, _set_etat);
    
    def _get_cliques_S_1(self) :
        """ Methode affichant le nombre de cliques couvrants un sommet . 
        """
        return self._cliques_S_1;
    
    def _set_cliques_S_1(self, nouveau_cliques_S_1) :
        """ Methode pour modifier le nom du sommet.
        """
        self._cliques_S_1 = nouveau_cliques_S_1;
    
    cliques_S_1 = property(_get_cliques_S_1, _set_cliques_S_1);
    
    
###############################################################################
#                   Classe 
###############################################################################