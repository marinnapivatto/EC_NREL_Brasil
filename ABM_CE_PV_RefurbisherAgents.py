# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent
"""


from mesa import Agent
import numpy as np


class Refurbishers(Agent):
    """
    A refurbisher which repairs modules, improve its processes and act as
    an intermediary between other actors

    Attributes:
        unique_id: agent #
        model (see ABM_CE_PV_Model)
        original_repairing_cost ($/Wp)
        original_repairing_volume (Wp)
        repairing_learning_shape_factor
        scndhand_mkt_pric_rate (ratio)

    """

    def __init__(self, unique_id, model, original_repairing_cost,
                 original_repairing_volume, repairing_learning_shape_factor,
                 scndhand_mkt_pric_rate):
        """
        Creation of new refurbisher agent
        """
        super().__init__(unique_id, model)
        self.original_repairing_cost = np.random.triangular(
            original_repairing_cost[0], original_repairing_cost[2],
            original_repairing_cost[1])
        self.original_repairing_volume = original_repairing_volume * 1E6
        self.repairing_cost = self.original_repairing_cost
        self.refurbished_volume = 0
        self.repairing_shape_factor = repairing_learning_shape_factor
        self.scndhand_mkt_pric_rate = np.random.triangular(
            scndhand_mkt_pric_rate[0], scndhand_mkt_pric_rate[2],
            scndhand_mkt_pric_rate[1])
        self.scd_hand_price = self.scndhand_mkt_pric_rate * \
                              self.model.fsthand_mkt_pric

    def repairable_volumes(self):
        """
        Compute amount of waste that can be repaired (and thus sold)
        """
        self.refurbished_volume = 0
        total_volume_recyler = 0
        for agent in self.model.schedule.agents:
            if self.model.num_consumers <= agent.unique_id < \
                    self.model.num_consumers + self.model.num_recyclers:
                total_volume_recyler += agent.repairable_volume
            if agent.unique_id < self.model.num_consumers:
                if agent.refurbisher_id == self.unique_id and \
                        agent.EoL_pathway == "repair":
                    self.refurbished_volume += agent.number_product_EoL
        self.refurbished_volume += total_volume_recyler / \
                                   self.model.num_refurbishers
        self.model.tot_refurb += self.refurbished_volume

    def refurbisher_learning_curve_function(self):
        """
        Run the learning curve function from recyclers with refurbishers
        parameters
        """
        for agent in self.model.schedule.agents:
            if agent.unique_id == self.model.num_consumers:
                self.repairing_cost = agent.learning_curve_function(
                    self.original_repairing_volume, self.refurbished_volume,
                    self.original_repairing_cost, self.repairing_shape_factor)

    def step(self):
        """
        Evolution of agent at each step
        """
        self.repairable_volumes()
        self.refurbisher_learning_curve_function()

