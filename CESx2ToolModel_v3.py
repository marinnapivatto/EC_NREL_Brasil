# -*- coding:utf-8 -*-
"""
Created on Wed Nov 21 09:33 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Model
"""
# TO DO: continue implementing model
# Add a procedure to add some agents!!!
# need to find a way to add some nods (and edge to the network)
# and place an agent on the node!!

from mesa import Model
from CESx2ToolAgents_v3 import Residential
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx  # Build network of agents
import matplotlib.pyplot as plt


class CESx2(Model):
    """
    Circular Economy Strategy for Clean Energy Simulation (CESx2)
    agent-based model based on various work (e.g., Meng et al. 2018
    and Ghali et al. 2017)

    Attributes:
        num_nodes
        avf_node_degree
        network_type=("small-world", "complete graph", "cycle graph",
            "scale-free graph")
        init_recycling_rate
        number_product (a list for the whole population e.g.
            (time series of product quantity))
        product_growth
        failure_rate_alpha
        trust_increment
        difficulty_recycling
        independence_level
        product_lifetime

    """

    def __init__(self, num_nodes=100, avg_node_degree=5,
                 network_type="small-world", init_recycling_rate=0.3,
                 total_number_product=[19, 19, 19, 19, 19, 19, 19, 58, 77, 157,
                                       246, 305, 496, 799, 1268, 2171, 2639,
                                       2227, 2357, 2488],
                 product_growth=[0.025, 0.09], failure_rate_alpha=5.4,
                 trust_increment=0, difficulty_recycling=0.57,
                 independence_level=0.5, product_lifetime=30):
        """
        Initiate model
        """
        self.num_nodes = num_nodes
        self.avg_node_degree = avg_node_degree
        self.network_type = network_type
        self.G = self.init_network()
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.init_recycling_rate = init_recycling_rate
        self.clock = 0
        self.total_number_product = total_number_product
        self.iteration = 0
        self.running = True
        self.color_map = []

        # Create agents, G nodes labels are equal to agents' unique_ID
        # Draw initial colored graph
        for i, node in enumerate(self.G.nodes()):
            a = Residential(i, self, product_growth, failure_rate_alpha,
                            trust_increment, difficulty_recycling,
                            independence_level, product_lifetime)
            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node)
        nx.draw(self.G, node_color=self.color_map, with_labels=True)
        plt.show()

        # Defines reporters and set up data collector
        CESx2_model_reporters = {
            "Recycling_count": lambda c: self.count_EoL("recycling"),
            "Landfilling_count": lambda c: self.count_EoL("landfilling"),
            "Product_stock_count": lambda c:
            self.count_stocks("product_stock"),
            "Product_recycled_count": lambda c:
            self.count_stocks("product_recycled"),
            "Product_landfilled_count":
                lambda c: self.count_stocks("product_landfilled")}

        CESx2_agent_reporters = {
            "Number_product_recycled":
                lambda a: getattr(a, "number_product_recycled", None),
            "Recycling":
                lambda a: getattr(a, "recycling", None)}

        self.datacollector = DataCollector(
            model_reporters=CESx2_model_reporters,
            agent_reporters=CESx2_agent_reporters)

    def init_network(self):
        if self.network_type == "small-world":
            return nx.watts_strogatz_graph(self.num_nodes,
                                               self.avg_node_degree, 0.1)
        elif self.network_type == "complete graph":
            return nx.complete_graph(self.num_nodes)
        elif self.network_type == "cycle graph":
            return nx.cycle_graph(self.num_nodes)
        elif self.network_type == "scale-free graph":
            return nx.powerlaw_cluster_graph(self.num_nodes,
                                             self.avg_node_degree, 0.1)
        else:
            return nx.watts_strogatz_graph(self.num_nodes,
                                               self.avg_node_degree, 0.1)

    def count_EoL(model, condition):
        count = 0
        for agent in model.schedule.agents:
            if condition == "recycling" and agent.recycling is True:
                count += 1
            elif condition == "landfilling" and agent.recycling is False:
                count += 1
            else:
                continue
        return count

    def count_stocks(model, condition):
        count = 0
        for agent in model.schedule.agents:
            if condition == "product_stock":
                count += sum(agent.number_product)
            elif condition == "product_recycled":
                count += agent.number_product_recycled
            elif condition == "product_landfilled":
                count += agent.number_product_landfilled
            else:
                continue
        return count

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        # following line refers to agent step function
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

