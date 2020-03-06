# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent
"""

from mesa import Agent
import numpy as np
import networkx as nx
import random


class Producers(Agent):
    """
    A producer which buys recycled materials, following the model from Ghali
    et al. 2017. The description of IS in Mathur et al.  2020 is also used.
    Prices of recycled materials are assumed to be competitive with the price
    of virgin materials. It is assumed that there is no volume threshold to
    Establish IS as recyclers may keep materials until they have enough to ship
    to producer. For materials like Al and glass, industries already
    incorporate recycled materials. Thus the market is established for those
    materials and the IS model from Ghali et al. 2017 is bypassed, considering
    those materials are directly reused in established markets. For materials
    like Si or Ag we assume other industries would accept recycled materials
    following the IS model from Ghali et al. 2017 (see Mathur et al. 2020).

    Attributes:
        unique_id: agent #
        model (see ABM_CE_PV_Model)
        scd_mat_prices ($/kg)
        social_influencability_boundaries (from Ghali et al. 2017)
        self_confidence_boundaries (from Ghali et al. 2017)
        product_average_wght (kg/Wp)

    """

    def __init__(self, unique_id, model, scd_mat_prices,
                 social_influencability_boundaries, self_confidence_boundaries,
                 product_average_wght):
        """
        Creation of new producer agent
        """
        super().__init__(unique_id, model)
        self.trust_history = np.copy(self.model.trust_prod)
        self.social_influencability = np.random.uniform(
            social_influencability_boundaries[0],
            social_influencability_boundaries[1])
        self.agent_i = self.unique_id - self.model.num_consumers
        self.knowledge = np.random.random()
        self.social_interactions = np.random.random()
        self.knowledge_learning = np.random.random()
        self.knowledge_t = self.knowledge
        self.acceptance = 0
        self.symbiosis = False
        self.self_confidence = np.random.uniform(self_confidence_boundaries[0],
                                                 self_confidence_boundaries[1])
        self.material_produced = random.choice(list(
            self.model.product_mass_fractions.keys()))
        self.recycled_material_volume = 0
        self.recycling_volume = 0
        self.recycled_mat_price = np.random.triangular(
            scd_mat_prices[self.material_produced][0], scd_mat_prices[
                self.material_produced][2], scd_mat_prices[
                self.material_produced][1])
        self.product_average_wght = product_average_wght
        self.recycled_material_value = 0

    def update_trust(self):
        """
        Update trust of agents in one another within the industrial symbiosis
        network. Mathematical model adapted from Ghali et al. 2017.
        """
        random_social_event = np.asmatrix(np.random.uniform(
            self.model.social_event_boundaries[0],
            self.model.social_event_boundaries[1], (
                self.model.num_prod_n_recyc, self.model.num_prod_n_recyc)))
        for agent in self.model.schedule.agents:
            if self.model.num_consumers <= agent.unique_id < \
                    self.model.num_consumers + self.model.num_prod_n_recyc:
                agent_j = agent.unique_id - self.model.num_consumers
                common_neighbors = list(
                    nx.common_neighbors(self.model.G, self.unique_id,
                                        agent.unique_id))
                if common_neighbors:
                    trust_neighbors = [self.model.trust_prod[self.agent_i, i -
                                                    self.model.num_consumers]
                                       for i in
                                       common_neighbors]
                    avg_trust_neighbors = self.social_influencability * (
                            sum(trust_neighbors) / len(trust_neighbors) -
                            self.trust_history[self.agent_i, agent_j])
                # Slight modification from Ghali et al.: if no common contact
                # there is no element for reputation
                else:
                    avg_trust_neighbors = 0
                trust_ij = self.trust_history[self.agent_i, agent_j] + \
                           avg_trust_neighbors + random_social_event[
                               self.agent_i, agent_j]
                if trust_ij < -1:
                    trust_ij = -1
                if trust_ij > 1:
                    trust_ij = 1
                self.model.trust_prod[self.agent_i, agent_j] = trust_ij
        self.trust_history = ((self.trust_history * (self.model.clock + 1)) +
                              self.model.trust_prod) / (self.model.clock + 2)

    def update_knowledge(self):
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

    def update_acceptance(self):
        """
        Update agents' acceptance of industrial symbiosis. Mathematical model
        adapted from Ghali et al. 2017.
        """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        neighbors_influence = len([agent for agent in
                                   self.model.grid.get_cell_list_contents(
                                       neighbors_nodes) if
                                   agent.symbiosis]) / len([agent for agent in
                                        self.model.grid.get_cell_list_contents(
                                        neighbors_nodes)])
        self.acceptance += self.social_influencability * neighbors_influence \
                           + self.self_confidence * \
                           (self.knowledge - self.knowledge_t)
        self.knowledge_t = self.knowledge
        if self.acceptance < 0:
            self.acceptance = 0
        if self.acceptance > 1:
            self.acceptance = 1

    def update_willingness(self):
        """
        Update willingness to form an industrial synergy. Mathematical
        model adapted from Ghali et al. 2017.
        """
        number_synergies = 0
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
            agent_j = agent.unique_id - self.model.num_consumers
            if self.model.trust_prod[self.agent_i, agent_j] >= \
                    self.model.trust_threshold and self.knowledge > \
                    self.model.knowledge_threshold:
                self.model.willingness[self.agent_i, agent_j] = self.acceptance
                number_synergies += 1
        if number_synergies > 0:
            self.symbiosis = True

    def recovered_volume_n_value(self):
        """
        Compute exchanged volumes from industrial synergy. Mathematical
        model adapted from Ghali et al. 2017.
        """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        for agent in self.model.grid.get_cell_list_contents(neighbors_nodes):
            agent_j = agent.unique_id - self.model.num_consumers
            if self.model.established_scd_mkt[self.material_produced]:
                self.recycled_material_volume += \
                    self.model.product_mass_fractions[
                        self.material_produced] * agent.recycling_volume * \
                    self.product_average_wght
            else:
                if self.model.willingness[self.agent_i, agent_j] >= \
                        self.model.willingness_threshold:
                    self.recycled_material_volume += \
                        self.model.product_mass_fractions[
                            self.material_produced] * agent.recycling_volume\
                        * self.product_average_wght
        self.recycled_material_value = self.recycled_mat_price * \
                                       self.recycled_material_volume

    def step(self):
        """
        Evolution of agent at each step
        """
        self.update_trust()
        self.update_knowledge()
        self.update_acceptance()
        self.update_willingness()
        self.recovered_volume_n_value()
