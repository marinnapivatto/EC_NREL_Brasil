# -*- coding:utf-8 -*-
"""
Created on Wed Nov 20 12:40 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Agent
"""


from mesa import Agent
import numpy as np
import random
from collections import OrderedDict
from scipy.stats import skewnorm


class Consumers(Agent):
    """
    A residential (or non-residential) owner of a product (e.g. PV,
    electronics) which dispose of it at its end of life and buy a first-hand
    or a second-hand product according to the Theory of Planned Behavior (TPB)

    Attributes:
        unique_id: agent #
        model (see ABM_CE_PV_Model)
        product_growth (a list for a piecewise function) (ratio)
        failure_rate_alpha (a list for a triangular distribution)
        perceived_behavioral_control (a list containing costs of each end of
            life (EoL) pathway)
        w_sn_eol (the weight of subjective norm in the agents' end of life
            (EoL) decisions as modeled with the theory of planned behavior)
        w_pbc_eol (the weight of perceived behavioral control in the agents'
            EoL decisions as modeled with the theory of planned behavior)
        w_a_eol (the weight of attitude in the agents' EoL decisions as modeled
            with the theory of planned behavior)
        w_sn_reuse (the weight of subjective norm in the agents' end of life
            purchase decisions as modeled with the theory of planned behavior)
        w_pbc_reuse (the weight of perceived behavioral control in the agents'
            purchase decisions as modeled with the theory of planned behavior)
        w_a_reuse (the weight of attitude in the agents' purchase decisions as
        modeled with the theory of planned behavior)
        product_lifetime (years)
        landfill_cost (a list for a triangular distribution) ($/Wp)
        hoarding_cost (a list for a triangular distribution) ($/Wp)

    """

    def __init__(self, unique_id, model, product_growth, failure_rate_alpha,
                 perceived_behavioral_control, w_sn_eol, w_pbc_eol, w_a_eol,
                 w_sn_reuse, w_pbc_reuse, w_a_reuse, product_lifetime,
                 landfill_cost, hoarding_cost):
        """
        Creation of new consumer agent
        """
        super().__init__(unique_id, model)
        self.breed = "residential"
        self.trust_levels = []
        self.number_product_EoL = 0
        self.number_product_repaired = 0
        self.number_product_sold = 0
        self.number_product_recycled = 0
        self.number_product_landfilled = 0
        self.number_product_hoarded = 0
        self.EoL_pathway = self.init_manage_EoL()
        self.number_product = [x / model.num_consumers * 1E6 for x
                               in model.total_number_product]
        self.product_growth_list = product_growth
        self.product_growth = self.product_growth_list[0]
        self.failure_rate_alpha = np.random.triangular(failure_rate_alpha[0],
                                                       failure_rate_alpha[2],
                                                       failure_rate_alpha[1])
        self.perceived_behavioral_control = perceived_behavioral_control
        self.w_sn_eol = w_sn_eol
        self.w_pbc_eol = w_pbc_eol
        self.w_a_eol = w_a_eol
        self.w_sn_reuse = w_sn_reuse
        self.w_pbc_reuse = w_pbc_reuse
        self.w_a_reuse = w_a_reuse
        self.product_lifetime = product_lifetime
        if self.EoL_pathway == "landfill" or self.EoL_pathway == "hoard":
            model.color_map.append('blue')
        else:
            model.color_map.append('green')
        self.recycling_facility_id = model.num_consumers + random.randrange(
            model.num_recyclers)
        self.refurbisher_id = model.num_consumers + model.num_prod_n_recyc + \
                              random.randrange(model.num_refurbishers)
        self.landfill_cost = np.random.triangular(landfill_cost[0],
                                                  landfill_cost[2],
                                                  landfill_cost[1])
        self.hoarding_cost = np.random.triangular(hoarding_cost[0],
                                                  hoarding_cost[2],
                                                  hoarding_cost[1])
        # Attitude level based on descriptive statistics for Pro-Environmental
        # Orientation level in the US in Saphores 2012. Skew normal
        # distribution are chosen to fit descriptive statistics. Attitude level
        # is bounded to [0, 1].
        self.attitude_level = \
            self.attitude_level_distribution(0, 0.6, 0.2)
        # Similarly, we use willingness to pay in Abbey et al. 2016 as a proxy
        # for attitude and fit the distribution to descriptive statistics.
        self.attitude_levels_pathways = [0] * len(self.model.all_EoL_pathways)
        self.attitude_level_reuse = \
            self.attitude_level_distribution(0.42, 0.5, 0.2)
        self.purchase_choices = ["new", "used"]
        self.purchase_choice = "new"
        self.attitude_levels_purchase = [0] * len(self.purchase_choices)
        self.pbc_reuse = [self.model.fsthand_mkt_pric, np.nan]
        self.second_choice = np.nan

    def attitude_level_distribution(self, a, loc, scale):
        """
        Distribute pro-environmental attitude level toward the decision in the
        population.
        """
        distribution = skewnorm(a, loc, scale)
        attitude_level = float(distribution.rvs(1))
        if attitude_level < 0:
            attitude_level = 0
        if attitude_level > 1:
            attitude_level = 1
        return attitude_level

    def init_manage_EoL(self):
        """
        Initiate the EoL pathway chosen by agents
        """
        if self.random.random() <= self.model.init_manageEoL_rate:
            return random.choice([x for x in self.model.all_EoL_pathways
                                  if (x == "recycle")])
        else:
            return random.choice([x for x in self.model.all_EoL_pathways
                                  if (x == "sell" or x == "repair" or
                                      x == "landfill" or x == "hoard")])

    def update_product_stock(self):
        """
        Update stock according to product growth and product failure
        Product failure is modeled with the Weibull function
        """
        self.number_product.append(self.number_product[-1] *
                                   (1 + self.product_growth))
        self.waste = self.model.waste_generation(self.failure_rate_alpha,
                                                 self.number_product)
        self.number_product_EoL = sum(self.waste)
        self.number_product = [product - waste for product, waste in
                               zip(self.number_product, self.waste)]

    def tpb_subjective_norm(self, decision, list_choices, weight_sn):
        """
        Calculate subjective norm (peer pressure) component of EoL TPB rule
        """
        neighbors_nodes = self.model.grid.get_neighbors(self.pos,
                                                        include_center=False)
        proportions_choices = []
        for i in range(len(list_choices)):
            proportion_choice = len([
                agent for agent in
                self.model.grid.get_cell_list_contents(neighbors_nodes) if
                getattr(agent, decision) == list_choices[i]]) / \
                                          len([agent for agent in
                                        self.model.grid.get_cell_list_contents(
                                                   neighbors_nodes)])
            proportions_choices.append(proportion_choice)
        return [weight_sn * x for x in proportions_choices]

    def tpb_perceived_behavioral_control(self, decision, pbc_choice,
                                         weight_pbc):
        """
        Calculate perceived behavioral control component of EoL TPB rule.
        Following Ghali et al. 2017 and Labelle et al. 2018, perceived
        behavioral control is understood as a function of financial costs.
        """
        max_cost = max(abs(i) for i in pbc_choice)
        pbc_choice = [i / max_cost for i in pbc_choice]
        if decision == "EoL_pathway":
            self.repairable_modules(pbc_choice)
        return [weight_pbc * -1 * max(i, 0) for i in pbc_choice]

    def tpb_attitude(self, decision, att_levels, att_level, weight_a):
        """
        Calculate pro-environmental attitude component of EoL TPB rule. Options
        considered pro environmental get a higher score than other options.
        """
        for i in range(len(att_levels)):
            if decision == "EoL_pathway":
                if self.model.all_EoL_pathways[i] == "repair" or \
                        self.model.all_EoL_pathways[i] == "sell" or \
                        self.model.all_EoL_pathways[i] == "recycle":
                    att_levels[i] = att_level
                else:
                    att_levels[i] = 1 - att_level
            elif decision == "purchase_choice":
                if self.purchase_choices[i] == "used":
                    att_levels[i] = att_level
                else:
                    att_levels[i] = 1 - att_level
        return [weight_a * x for x in att_levels]

    def yearly_waste(self):
        """
        Update total waste generated
        """
        self.model.total_waste += self.number_product_EoL

    def repairable_modules(self, pbc_choice):
        """
        Account for the fact that some panels cannot be repaired
        (and thus sold)
        """
        total_waste = 0
        sold_waste = 0
        total_volume_refurbished = 0
        for agent in self.model.schedule.agents:
            if self.model.num_consumers + self.model.num_prod_n_recyc <= \
                    agent.unique_id:
                total_volume_refurbished += agent.refurbished_volume
            if agent.unique_id < self.model.num_consumers:
                total_waste += agent.number_product_EoL
                if agent.EoL_pathway == "sell":
                    sold_waste += agent.number_product_EoL
        if sold_waste + total_volume_refurbished > \
                self.model.repairability * total_waste:
            pbc_choice[0] = 1
            pbc_choice[1] = 1

    def eol_choice(self, decision, list_choices, weight_sn, pbc_choice,
                   weight_pbc, att_levels, att_level, weight_a):
        """
        Select the EoL pathway with highest behavioral intention. Behavioral
        intention is a function of the subjective norm, the perceived
        behavioral control and attitude.
        """
        sn_values = self.tpb_subjective_norm(
            decision, list_choices, weight_sn)
        pbc_values = self.tpb_perceived_behavioral_control(
            decision, pbc_choice, weight_pbc)
        a_values = self.tpb_attitude(decision, att_levels, att_level, weight_a)
        self.behavioral_intentions = [(pbc_values[i]) + sn_values[i] +
                                      a_values[i] for i in
                                      range(len(pbc_values))]
        self.pathways_and_BI = {list_choices[i]: self.behavioral_intentions[i]
                                for i in
                                range(len(list_choices))}
        shuffled_dic = list(self.pathways_and_BI.items())
        random.shuffle(shuffled_dic)
        self.pathways_and_BI = OrderedDict(shuffled_dic)
        first_choice = np.nan
        for key, value in self.pathways_and_BI.items():
            if value == np.nan:
                return self.EoL_pathway
            else:
                max_bi = max(self.behavioral_intentions)
                second_choice = list(self.behavioral_intentions)
                second_choice.remove(max_bi)
                if value == max(second_choice):
                    self.second_choice = key
                if value == max(self.behavioral_intentions):
                    first_choice = key
        return first_choice

    def volume_used_products_purchased(self):
        """
        Count amount of remanufactured product that are bought by consumers
        """
        self.purchase_choice = \
            self.eol_choice("purchase_choice", self.model.purchase_options,
                            self.w_sn_reuse, self.pbc_reuse, self.w_pbc_reuse,
                            self.attitude_levels_purchase,
                            self.attitude_level_reuse, self.w_a_reuse)
        if self.purchase_choice == "used":
            self.model.volume_used_purchased += self.number_product_EoL

    def update_product_eol(self):
        """
        The amount of waste generated is taken from "update_product_stock"
        and attributed to the chosen EoL pathway
        """
        self.EoL_pathway = \
            self.eol_choice("EoL_pathway", self.model.all_EoL_pathways,
                            self.w_sn_eol, self.perceived_behavioral_control,
                            self.w_pbc_eol, self.attitude_levels_pathways,
                            self.attitude_level, self.w_a_eol)
        if self.EoL_pathway == "repair":
            self.number_product_repaired = self.number_product_repaired + \
                                           self.number_product_EoL
        elif self.EoL_pathway == "sell":
            self.number_product_sold = self.number_product_sold + \
                                       self.number_product_EoL
        elif self.EoL_pathway == "recycle":
            self.number_product_recycled = self.number_product_recycled + \
                                           self.number_product_EoL
        elif self.EoL_pathway == "landfill":
            self.number_product_landfilled = self.number_product_landfilled + \
                                             self.number_product_EoL
        else:
            self.number_product_hoarded = self.number_product_hoarded + \
                                          self.number_product_EoL

    def update_perceived_behavioral_control(self):
        """
        Costs from each EoL pathway and related perceived behavioral control
        are updated according to processes from other agents or own initiated
        costs
        """
        for agent in self.model.schedule.agents:
            if agent.unique_id == self.recycling_facility_id:
                self.perceived_behavioral_control[2] = \
                    agent.recycling_cost
            elif agent.unique_id == self.refurbisher_id:
                self.perceived_behavioral_control[0] = \
                    agent.repairing_cost
                self.perceived_behavioral_control[1] = -1 * \
                                                       agent.scd_hand_price
                self.pbc_reuse[1] = agent.scd_hand_price
        self.perceived_behavioral_control[3] = self.landfill_cost
        self.perceived_behavioral_control[4] = self.hoarding_cost

    def step(self):
        """
        Evolution of agent at each step
        """
        # Update product growth from a list:
        if self.model.clock > 9:
            self.product_growth = self.product_growth_list[1]
        self.update_product_stock()
        self.yearly_waste()
        self.update_perceived_behavioral_control()
        self.volume_used_products_purchased()
        self.update_product_eol()
