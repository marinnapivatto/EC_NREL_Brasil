# -*- coding:utf-8 -*-
"""
Created on Wed Nov 21 09:33 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Model
"""

from mesa import Model
from ABM_CE_PV_ConsumerAgents import Consumers
from ABM_CE_PV_RecyclerAgents import Recyclers
from ABM_CE_PV_RefurbisherAgents import Refurbishers
from ABM_CE_PV_ProducerAgents import Producers
from mesa.time import RandomActivation
from mesa.time import BaseScheduler
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx  # Build network of agents
import numpy as np
from math import *
import matplotlib.pyplot as plt
import pandas as pd


class ABM_CE_PV(Model):
    """
    Intro:
    The agent-based simulations of the circular economy (ABSiCE) tool is an
    agent-based model based on various work (e.g., Tong et al. 2018
    and Ghali et al. 2017, IRENA-IEA 2016...). This instance of the model
    simulate the implementation of circular economy (CE) strategies.

    Purpose:
    ABSiCE aims at simulating the adoption of various various CE strategies.
    More specifically, the model enables exploring the effect of various policy
    or techno-economic factors on the adoption of the CE strategies by a
    system's actors.

    Attributes:
        num_consumers
        consumers_node_degree
        consumers_network_type=("small-world", "complete graph", "random"
            "cycle graph", "scale-free graph")
        num_recyclers
        num_producers
        prod_n_recyc_node_degree
        prod_n_recyc_network_type
        num_refurbishers
        init_manageEoL_rate (ratio)
        total_number_product (a list for the whole population e.g.
            (time series of product quantity)) (MWp)
        product_growth (a list for a piecewise function) (ratio)
        failure_rate_alpha (a list for a triangular distribution)
        hoarding_cost (a list for a triangular distribution) ($/Wp)
        landfill_cost (a list for a triangular distribution) ($/Wp)
        w_sn_eol (the weight of subjective norm in the agents' decisions as
            modeled with the theory of planned behavior)
        w_pbc_eol (the weight of perceived behavioral control in the agents'
            decisions as modeled with the theory of planned behavior)
        w_a_eol (the weight of attitude in the agents' decisions as modeled
            with the theory of planned behavior)
        w_sn_reuse (same as above but for remanufactured product purchase
            decision)
        w_pbc_reuse (same as above but for remanufactured product purchase
            decision)
        w_a_reuse (same as above but for remanufactured product purchase
            decision)
        product_lifetime (years)
        all_EoL_pathways (list of strings)
        original_recycling_cost (a list for a triangular distribution) ($/Wp)
        original_fraction_recycled_waste (ratio)
        recycling_learning_shape_factor
        repairability (ratio)
        original_repairing_cost (a list for a triangular distribution) ($/Wp)
        original_repairing_volume (Wp)
        repairing_learning_shape_factor
        scndhand_mkt_pric_rate (a list for a triangular distribution) (ratio)
        fsthand_mkt_pric ($/Wp)
        init_trust_boundaries (from Ghali et al. 2017)
        social_event_boundaries (from Ghali et al. 2017)
        social_influencability_boundaries (from Ghali et al. 2017)
        trust_threshold (from Ghali et al. 2017)
        knowledge_threshold (from Ghali et al. 2017)
        willingness_threshold (from Ghali et al. 2017)
        self_confidence_boundaries (from Ghali et al. 2017)
        product_mass_fractions (dictionary containing mass fractions of
            materials composing the product)
        established_scd_mkt (dictionary containing booleans regarding the
            availability of an industrial pathway for the recovered material)
        scd_mat_prices (dictionary containing lists for triangular
            distributions of secondary materials prices) ($/kg)
        product_average_wght (kg/Wp)
        recycling_states (a list of states owning at least one recycling
            facility)
        transportation_cost ($/t.km)

    """

    def __init__(self, num_consumers=100,
                 consumers_node_degree=5,
                 consumers_network_type="small-world",
                 num_recyclers=16,
                 num_producers=60,
                 prod_n_recyc_node_degree=5,
                 prod_n_recyc_network_type="small-world",
                 num_refurbishers=15,
                 # Saphores et al. 2012 e-waste recycling=0.14
                 # IRENA report PV recycling=0.3?
                 init_manageEoL_rate=0.14,
                 total_number_product=[44, 44, 44, 44, 44, 44, 44, 160, 289,
                                       435, 849, 1920, 3374, 4766, 6244, 7500,
                                       15128, 10608, 10672, 11774],
                 product_growth=[0.0892, 0.025],
                 failure_rate_alpha=[2.4928, 5.3759, 3.93495],
                 # EoL cost in $/Wp
                 hoarding_cost=[0, 0.01, 0.005],
                 landfill_cost=[0.003, 0.009, 0.006],
                 # Values from meta-analysis: Geiger et al. 2019 (eol) (
                 # sn 0.33, pbc 0.39, a 0.34 and Singhal et al. 2019 (reuse) (
                 # sn 0.497, pbc 0.382, a 0.464)
                 w_sn_eol=0.33,
                 w_pbc_eol=0.39,
                 w_a_eol=0.34,
                 w_sn_reuse=0.497,
                 w_pbc_reuse=0.382,
                 w_a_reuse=0.464,
                 product_lifetime=30,
                 all_EoL_pathways=["repair", "sell", "recycle", "landfill",
                               "hoard"],
                 # original_recycling_cost=[0.027, 0.128, 0.077]
                 # third_recycling_cost=[0.009, 0.042, 0.025]
                 original_recycling_cost=[0.027, 0.128, 0.077],
                 original_fraction_recycled_waste=0.14,
                 recycling_learning_shape_factor=-0.39,
                 repairability=0.55,
                 original_repairing_cost=[0.015, 0.208, 0.112],
                 original_repairing_volume=3000,
                 repairing_learning_shape_factor=-0.31,
                 scndhand_mkt_pric_rate=[0.4, 1, 0.7],
                 # New module price in $/Wp
                 fsthand_mkt_pric=0.3,
                 init_trust_boundaries=[-1, 1],
                 social_event_boundaries=[-1, 1],
                 social_influencability_boundaries=[0, 1],
                 # Assumed value of trust, knowledge threshold:
                 trust_threshold=0.5,
                 knowledge_threshold=0.5,
                 willingness_threshold=0.5,
                 self_confidence_boundaries=[0, 1],
                 # Assumed Al, glass cullet and silver for PV case study
                 product_mass_fractions={"Aluminum": 0.18, "Glass": 0.69,
                                         "Silver": 0.005},
                 # Some industries already incorporate recycled materials. Thus
                 # the market is established for those materials (designated by
                 # True).
                 established_scd_mkt={"Aluminum": True, "Glass": True,
                                      "Silver": False},
                 # Secondary materials prices in $/kg
                 scd_mat_prices={"Aluminum": [0.66, 1.98, 1.32], "Glass":
                     [0.01, 0.06, 0.035], "Silver": [453, 653, 582]},
                 # Weight in kg/Wp
                 product_average_wght=0.077,
                 # List of states with at least one recycling facility
                 recycling_states=['Texas', 'Arizona', 'Oregon', 'Oklahoma',
                                   'Wisconsin', 'Ohio', 'Kentucky',
                                   'South Carolina'],
                 # Transportation costs in $/t.km
                 transportation_cost=0.021761):
        """
        Initiate model
        """
        # Set up variables
        self.num_consumers = num_consumers
        self.consumers_node_degree = consumers_node_degree
        self.consumers_network_type = consumers_network_type
        self.num_recyclers = num_recyclers
        self.num_prod_n_recyc = num_recyclers + num_producers
        self.prod_n_recyc_node_degree = prod_n_recyc_node_degree
        self.prod_n_recyc_network_type = prod_n_recyc_network_type
        self.num_refurbishers = num_refurbishers
        self.H1 = self.init_network(self.consumers_network_type,
                                   self.num_consumers,
                                   self.consumers_node_degree)
        self.H2 = self.init_network(self.prod_n_recyc_network_type,
                                   self.num_prod_n_recyc,
                                   self.prod_n_recyc_node_degree)
        self.H3 = self.init_network("complete graph", self.num_refurbishers,
                                    "NaN")
        self.G = nx.disjoint_union(self.H1, self.H2)
        self.G = nx.disjoint_union(self.G, self.H3)
        self.grid = NetworkGrid(self.G)
        self.schedule = BaseScheduler(self)
        self.init_manageEoL_rate = init_manageEoL_rate
        self.clock = 0
        self.total_number_product = total_number_product
        self.iteration = 0
        self.running = True
        self.color_map = []
        self.all_EoL_pathways = all_EoL_pathways
        self.purchase_options = ["new", "used"]
        self.avg_failure_rate = failure_rate_alpha
        self.original_num_prod = total_number_product
        self.avg_lifetime = product_lifetime
        self.fsthand_mkt_pric = fsthand_mkt_pric
        self.repairability = repairability
        self.total_waste = 0
        self.volume_used_purchased = 0
        self.sold_waste = 0
        self.recycler_repairable_waste = 0
        self.tot_refurb = 0
        perceived_behavioral_control = [np.nan] * len(all_EoL_pathways)

        # Adjacency matrix of trust network: trust of row index into column
        self.trust_prod = np.asmatrix(np.random.uniform(
            init_trust_boundaries[0], init_trust_boundaries[1],
            (self.num_prod_n_recyc, self.num_prod_n_recyc)))
        np.fill_diagonal(self.trust_prod, 0)
        self.social_event_boundaries = social_event_boundaries
        self.trust_threshold = trust_threshold
        self.knowledge_threshold = knowledge_threshold
        self.willingness_threshold = willingness_threshold
        self.willingness = np.asmatrix(np.zeros((self.num_prod_n_recyc,
                                                self.num_prod_n_recyc)))
        self.product_mass_fractions = product_mass_fractions
        self.established_scd_mkt = established_scd_mkt

        # Define recycling states and find average transportation distance
        self.states = pd.read_csv("StatesAdjacencyMatrix.csv").to_numpy()
        self.states_graph = nx.from_numpy_matrix(self.states)
        nodes_states_dic = \
            dict(zip(list(self.states_graph.nodes),
                     list(pd.read_csv("StatesAdjacencyMatrix.csv"))))
        self.states_graph = nx.relabel_nodes(self.states_graph,
                                             nodes_states_dic)
        self.recycling_states = recycling_states
        distances_to_recyclers = []
        for i in self.states_graph.nodes:
            shortest_paths = []
            for j in self.recycling_states:
                shortest_paths.append(
                    nx.shortest_path_length(self.states_graph, source=i,
                                            target=j, weight='weight',
                                            method='dijkstra'))
            shortest_paths_closest_recycler = min(shortest_paths)
            distances_to_recyclers.append(shortest_paths_closest_recycler)
        self.mn_mx_av_distance_to_recycler = [
            min(distances_to_recyclers), max(distances_to_recyclers),
            sum(distances_to_recyclers) / len(distances_to_recyclers)]
        self.transportation_cost = [
            x * transportation_cost / 1E3 * product_average_wght for x in
            self.mn_mx_av_distance_to_recycler]
        original_recycling_cost = [sum(x) for x in zip(
            original_recycling_cost, self.transportation_cost)]

        # Create agents, G nodes labels are equal to agents' unique_ID
        for node in self.G.nodes():
            if node < self.num_consumers:
                a = Consumers(node, self, product_growth, failure_rate_alpha,
                              perceived_behavioral_control, w_sn_eol,
                              w_pbc_eol, w_a_eol, w_sn_reuse, w_pbc_reuse,
                              w_a_reuse, product_lifetime, landfill_cost,
                              hoarding_cost)
                self.schedule.add(a)
                # Add the agent to the node
                self.grid.place_agent(a, node)
            elif node < self.num_recyclers + self.num_consumers:
                b = Recyclers(node, self, original_recycling_cost,
                              original_fraction_recycled_waste,
                              recycling_learning_shape_factor,
                              social_influencability_boundaries)
                self.schedule.add(b)
                self.grid.place_agent(b, node)
            elif node < self.num_prod_n_recyc + self.num_consumers:
                c = Producers(node, self, scd_mat_prices,
                              social_influencability_boundaries,
                              self_confidence_boundaries, product_average_wght)
                self.schedule.add(c)
                self.grid.place_agent(c, node)
            else:
                d = Refurbishers(node, self, original_repairing_cost,
                                 original_repairing_volume,
                                 repairing_learning_shape_factor,
                                 scndhand_mkt_pric_rate)
                self.schedule.add(d)
                self.grid.place_agent(d, node)
        # Draw initial graph
        # nx.draw(self.G, with_labels=True)
        # plt.show()

        # Defines reporters and set up data collector
        ABM_CE_PV_model_reporters = {
            "Year": lambda c:
            self.report_output("year"),
            "Agents repairing": lambda c: self.count_EoL("repairing"),
            "Agents selling": lambda c: self.count_EoL("selling"),
            "Agents recycling": lambda c: self.count_EoL("recycling"),
            "Agents landfilling": lambda c: self.count_EoL("landfilling"),
            "Agents storing": lambda c: self.count_EoL("hoarding"),
            "In operation": lambda c:
            self.report_output("product_stock"),
            "End-of-life - repaired": lambda c:
            self.report_output("product_repaired"),
            "End-of-life - sold": lambda c:
            self.report_output("product_sold"),
            "End-of-life - recycled": lambda c:
            self.report_output("product_recycled"),
            "End-of-life - landfilled": lambda c:
            self.report_output("product_landfilled"),
            "End-of-life - stored": lambda c:
            self.report_output("product_hoarded"),
            "Average landfilling cost": lambda c:
            self.report_output("average_landfill_cost"),
            "Average storing cost": lambda c:
            self.report_output("average_hoarding_cost"),
            "Average recycling cost": lambda c:
            self.report_output("average_recycling_cost"),
            "Average repairing cost": lambda c:
            self.report_output("average_repairing_cost"),
            "Average selling cost": lambda c:
            self.report_output("average_second_hand_price"),
            "Recycled material volume": lambda c:
            self.report_output("recycled_mat_volume"),
            "Recycled material value": lambda c:
            self.report_output("recycled_mat_value")}

        ABM_CE_PV_agent_reporters = {
            "Year": lambda c:
            self.report_output("year"),
            "Number_product_repaired":
                lambda a: getattr(a, "number_product_repaired", None),
            "Number_product_sold":
                lambda a: getattr(a, "number_product_sold", None),
            "Number_product_recycled":
                lambda a: getattr(a, "number_product_recycled", None),
            "Number_product_landfilled":
                lambda a: getattr(a, "number_product_landfilled", None),
            "Number_product_hoarded":
                lambda a: getattr(a, "number_product_hoarded", None),
            "Recycling":
                lambda a: getattr(a, "EoL_pathway", None),
            "Landfilling costs":
                lambda a: getattr(a, "landfill_cost", None),
            "Storing costs":
                lambda a: getattr(a, "hoarding_cost", None),
            "Recycling costs":
                lambda a: getattr(a, "recycling_cost", None),
            "Repairing costs":
                lambda a: getattr(a, "repairing_cost", None),
            "Selling costs":
                lambda a: getattr(a, "scd_hand_price", None),
            "Material produced":
                lambda a: getattr(a, "material_produced", None),
            "Recycled volume":
                lambda a: getattr(a, "recycled_material_volume", None),
            "Recycled value":
                lambda a: getattr(a, "recycled_material_value", None)}

        self.datacollector = DataCollector(
            model_reporters=ABM_CE_PV_model_reporters,
            agent_reporters=ABM_CE_PV_agent_reporters)

    def init_network(self, network, nodes, node_degree):
        """
        Set up model's industrial symbiosis (IS) and consumers networks
        """
        if network == "small-world":
            return nx.watts_strogatz_graph(nodes, node_degree, 0.1)
        elif network == "complete graph":
            return nx.complete_graph(nodes)
        if network == "random":
            return nx.watts_strogatz_graph(nodes, node_degree, 1)
        elif network == "cycle graph":
            return nx.cycle_graph(nodes)
        elif network == "scale-free graph":
            return nx.powerlaw_cluster_graph(nodes, node_degree, 0.1)
        else:
            return nx.watts_strogatz_graph(nodes, node_degree, 0.1)

    def waste_generation(self, failure_rate, num_product):
        """
        Generate waste, called by consumers and recyclers (to get original
        recycling amount)
        """
        return [j * (1 - e**(-((self.clock + (19 - z)) /
                        self.avg_lifetime)**failure_rate)).real
                      for (z, j) in enumerate(num_product)]

    def count_EoL(model, condition):
        """
        Count adoption in each end of life pathway. Values are then
        reported by model's reporters.
        """
        count = 0
        for agent in model.schedule.agents:
            if agent.unique_id < model.num_consumers:
                if condition == "repairing" and agent.EoL_pathway == "repair":
                    count += 1
                elif condition == "selling" and agent.EoL_pathway == "sell":
                    count += 1
                elif condition == "recycling" and \
                        agent.EoL_pathway == "recycle":
                    count += 1
                elif condition == "landfilling" and \
                        agent.EoL_pathway == "landfill":
                    count += 1
                elif condition == "hoarding" and agent.EoL_pathway == "hoard":
                    count += 1
                else:
                    continue
            else:
                continue
        return count

    def report_output(model, condition):
        """
        Count waste streams in each end of life pathway. Values are then
        reported by model's reporters.
        """
        count = 0
        repairable_volume_recyclers = 0
        for agent in model.schedule.agents:
            if model.num_consumers <= agent.unique_id < \
                    model.num_consumers + model.num_recyclers:
                repairable_volume_recyclers += agent.total_repairable_volume
        repairable_volume_recyclers = repairable_volume_recyclers / \
                                      model.num_consumers
        for agent in model.schedule.agents:
            if condition == "product_stock" and agent.unique_id < \
                    model.num_consumers:
                count += sum(agent.number_product)
            elif condition == "product_repaired" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_repaired
            elif condition == "product_sold" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_sold
                count += repairable_volume_recyclers
            elif condition == "product_recycled" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_recycled
                count -= repairable_volume_recyclers
            elif condition == "product_landfilled" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_landfilled
            elif condition == "product_hoarded" and agent.unique_id < \
                    model.num_consumers:
                count += agent.number_product_hoarded
            elif condition == "average_landfill_cost" and agent.unique_id < \
                    model.num_consumers:
                count += agent.landfill_cost / model.num_consumers
            elif condition == "average_hoarding_cost" and agent.unique_id < \
                    model.num_consumers:
                count += agent.hoarding_cost / model.num_consumers
            elif condition == "average_recycling_cost" and model.num_consumers\
                    <= agent.unique_id < model.num_consumers + \
                    model.num_recyclers:
                count += agent.recycling_cost / model.num_recyclers
            elif condition == "average_repairing_cost" and model.num_consumers\
                    + model.num_prod_n_recyc <= agent.unique_id:
                count += agent.repairing_cost / model.num_refurbishers
            elif condition == "average_second_hand_price" and \
                    model.num_consumers + model.num_prod_n_recyc <= \
                    agent.unique_id:
                count += (-1 * agent.scd_hand_price) / model.num_refurbishers
            elif condition == "year":
                count = 2020 + model.clock
            elif condition == "recycled_mat_volume" and model.num_consumers + \
                    model.num_recyclers <= agent.unique_id < \
                    model.num_consumers + model.num_prod_n_recyc:
                count += agent.recycled_material_volume
            elif condition == "recycled_mat_value" and model.num_consumers + \
                    model.num_recyclers <= agent.unique_id < \
                    model.num_consumers + model.num_prod_n_recyc:
                count += agent.recycled_material_value
        return count

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        self.tot_refurb = 0
        self.sold_waste = 0
        self.recycler_repairable_waste = 0
        self.total_waste = 0
        # Collect data
        self.datacollector.collect(self)
        # Refers to agent step function
        self.schedule.step()
        self.clock = self.clock + 1
