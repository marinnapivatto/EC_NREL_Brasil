# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent
"""
# TO DO: continue implementing agents behavioral rules
# Difficulty can be drawn from Byrka et al. 2016 --> 0.57
# (see excel file DifficultyOfRecycling in:
# C:\Users\jwalzber\Documents\Fall19\Modeling_CESx2
# At the end put initial values in class to have a model working without
# specifying anything
# number_product=[19, 19, 19, 19, 19, 19, 19, 58, 77, 157, 246,
#                                305, 496, 799, 1268, 2171, 2639, 2227, 2357,
#                                2488]
# add a way to increase the number of agents
# fix all landfilled count !!!

from mesa import Agent
from math import *


class Residential(Agent):
    """
    A residential owner of a product (e.g. PV, electronics)
    which needs to dispose of it at its end of life

    Attributes:
        unique_id: agent #
        product_growth
        failure_rate_alpha
        trust_increment
        difficulty_recycling
        independence_level
        product_lifetime

    """

    def __init__(self, unique_id, model, product_growth, failure_rate_alpha,
                 trust_increment, difficulty_recycling, independence_level,
                 product_lifetime):
        """
        Creation of new agent
        """
        super().__init__(unique_id, model)
        self.breed = "residential"
        self.trust_levels = []
        self.number_product_EoL = 0
        self.number_product_landfilled = 0
        self.number_product_recycled = 0
        self.recycling_intent = self.model.init_recycling_rate
        self.recycling = self.init_recycling()
        self.number_product = [x / model.num_nodes for x
                               in model.total_number_product]
        self.product_growth_list = product_growth
        self.product_growth = self.product_growth_list[0]
        self.failure_rate_alpha = failure_rate_alpha
        self.trust_increment = trust_increment
        self.difficulty_recycling = difficulty_recycling
        self.independence_level = independence_level
        self.product_lifetime = product_lifetime
        if self.recycling:
            model.color_map.append('green')
        else:
            model.color_map.append('blue')

    def init_recycling(self):
        if self.random.random() <= self.model.init_recycling_rate:
            return True
        else:
            return False

    def update_product_stock(self):
        """
        Update stock according to product growth and product failure
        Product failure is modeled with the Weibull function
        """
        self.number_product.append(self.number_product[-1] *
                                   (1 + self.product_growth))
        self.waste = [j * (1 - e**(-((self.model.clock + (19 - z)) /
                        self.product_lifetime)**self.failure_rate_alpha)).real
                      for (z, j) in enumerate(self.number_product)
                      if j >= 0]
        self.number_product_EoL = sum(self.waste)
        self.number_product = [product - waste for product, waste in
                               zip(self.number_product, self.waste)]

    def social_norm_recycling(self):
        """
        Calculate peer pressure component of recycling rule
        """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        proportion_recycling = len([
            agent for agent in
            self.model.grid.get_cell_list_contents(neighbors_nodes) if
            agent.breed is "residential" and
            agent.recycling is True]) / len([agent for agent in
                    self.model.grid.get_cell_list_contents(neighbors_nodes) if
                                             agent.breed is "residential"])
        self.peer_pressure = (1 - self.independence_level) * \
                             proportion_recycling

    def recycling_choice(self):
        if (self.peer_pressure + (1 - self.difficulty_recycling) >=
                (1-self.recycling_intent)):
            self.recycling = True
        else:
            self.recycling = False
        self.recycling_intent = self.peer_pressure + \
                                (1 - self.difficulty_recycling)

    def update_product_EoL(self):
        if self.recycling:
            self.number_product_recycled = self.number_product_recycled +\
                                           self.number_product_EoL
        else:
            self.number_product_landfilled = self.number_product_landfilled + \
                                            self.number_product_EoL

    def step(self):
        """
        Evolution of agent at each step
        """
        # Update product growth from a list:
        # if self.model.clock > 11:
        #   self.product_growth = self.product_growth_list[1]
        self.update_product_stock()
        self.social_norm_recycling()
        self.recycling_choice()
        self.update_product_EoL()
