# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent
"""

from mesa import Agent
import numpy as np


class Recyclers(Agent):
    """
    A recycler which sells recycled materials and improve its processes

    Attributes:
        unique_id: agent #
        model (see ABM_CE_PV_Model)
        original_recycling_cost ($/Wp)
        original_fraction_recycled_waste (ratio)
        recycling_learning_shape_factor
        social_influencability_boundaries (from Ghali et al. 2017)

    """

    def __init__(self, unique_id, model, original_recycling_cost,
                 original_fraction_recycled_waste,
                 recycling_learning_shape_factor,
                 social_influencability_boundaries):
        """
        Creation of new recycler agent
        """
        super().__init__(unique_id, model)
        self.original_recycling_cost = np.random.triangular(
            original_recycling_cost[0], original_recycling_cost[2],
            original_recycling_cost[1])
        self.original_fraction_recycled_waste = \
            original_fraction_recycled_waste
        self.recycling_learning_shape_factor = recycling_learning_shape_factor
        self.recycling_cost = self.original_recycling_cost
        self.recycler_total_volume = 0
        self.recycling_volume = 0
        self.repairable_volume = 0
        self.total_repairable_volume = 0
        #  Original recycling volume is based on previous years EoL volume
        # (from 2009 to 2019)
        original_recycled_volumes = [x / model.num_recyclers * 1E6 for x
                                     in model.original_num_prod]
        self.original_recycling_volume = \
            (1 - self.model.repairability) * \
            self.original_fraction_recycled_waste * \
            sum(self.model.waste_generation(self.model.avg_failure_rate[1],
                                            original_recycled_volumes))
        self.social_influencability = np.random.uniform(
            social_influencability_boundaries[0],
            social_influencability_boundaries[1])
        self.knowledge = np.random.random()
        self.social_interactions = np.random.random()
        self.knowledge_learning = np.random.random()
        self.knowledge_t = self.knowledge
        self.symbiosis = False
        self.agent_i = self.unique_id - self.model.num_consumers

    def triage(self):
        """
        Evaluate amount of solar panels that can be refurbished
        """
        self.recycler_total_volume = 0
        self.recycling_volume = 0
        self.repairable_volume = 0
        for agent in self.model.schedule.agents:
            if agent.unique_id < self.model.num_consumers and \
                    agent.recycling_facility_id == self.unique_id and \
                    agent.EoL_pathway == "recycle":
                self.recycler_total_volume += agent.number_product_EoL
                if self.model.recycler_repairable_waste < \
                        self.model.repairability * self.model.total_waste:
                    self.recycling_volume = (1 - self.model.repairability) * \
                                            self.recycler_total_volume
                    self.repairable_volume = self.recycler_total_volume - \
                                             self.recycling_volume
                else:
                    self.recycling_volume = self.recycler_total_volume
                    self.repairable_volume = 0
        self.model.recycler_repairable_waste += self.repairable_volume
        self.total_repairable_volume += self.repairable_volume

    def learning_curve_function(self, original_volume, volume, original_cost,
                                shape_factor):
        """
        Account for the learning effect: recyclers and refurbishers improve
        their recycling and repairing processes respectively
        """
        if volume > 0:
            potential_recycling_cost = original_cost * \
                                       (volume / original_volume) ** \
                                       shape_factor
            if potential_recycling_cost < original_cost:
                return potential_recycling_cost
            else:
                return original_cost
        return original_cost

    def update_recyclers_knowledge(self):
        """
        Update knowledge of agents about industrial symbiosis. Mathematical
        model adapted from Ghali et al. 2017.
        """
        self.knowledge_learning = np.random.random()
        knowledge_neighbors = 0
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
            self.social_interactions = np.random.random()
            agent_j = agent.unique_id - self.model.num_consumers
            if self.model.trust_prod[self.agent_i, agent_j] >= \
                    self.model.trust_threshold:
                knowledge_neighbors += self.social_interactions * (
                        agent.knowledge - self.knowledge)
        self.knowledge_t = self.knowledge
        self.knowledge += self.social_influencability * knowledge_neighbors + \
                          self.knowledge_learning
        if self.knowledge < 0:
            self.knowledge = 0
        if self.knowledge > 1:
            self.knowledge = 1

    def step(self):
        """
        Evolution of agent at each step
        """
        self.triage()
        self.recycling_cost = self.learning_curve_function(
            self.original_recycling_volume, self.recycling_volume,
            self.original_recycling_cost,
            self.recycling_learning_shape_factor)
        self.update_recyclers_knowledge()
