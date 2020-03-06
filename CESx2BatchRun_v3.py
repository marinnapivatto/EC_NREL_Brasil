# -*- coding:utf-8 -*-
"""
Created on Wed Nov 21 12:43 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Run
"""

from ABM_CE_PV_Model import *
from mesa.batchrunner import BatchRunner


# Batch run model
fixed_params = {
    "num_consumers": 100,
    "consumers_node_degree": 5,
    "num_recyclers": 16,
    "num_producers": 60,
    "prod_n_recyc_node_degree": 5,
    "prod_n_recyc_network_type": "small-world",
    "num_refurbishers": 15,
    "total_number_product": [44, 44, 44, 44, 44, 44, 44, 160, 289,
                                       435, 849, 1920, 3374, 4766, 6244, 7500,
                                       15128, 10608, 10672, 11774],
    "product_growth": [0.0892, 0.025],
    "failure_rate_alpha": [2.4928, 5.3759, 3.93495],
    "hoarding_cost": [0, 0.01, 0.005],
    "landfill_cost": [0.003, 0.009, 0.006],
    "product_lifetime": 30,
    "all_EoL_pathways": ["repair", "sell", "recycle", "landfill",
                               "hoard"],
    "original_recycling_cost": [0.027, 0.128, 0.077],
    "original_fraction_recycled_waste": 0.3,
    "original_repairing_cost": [0.015, 0.208, 0.112],
    "original_repairing_volume": 3000,
    "repairing_learning_shape_factor": -0.31,
    "scndhand_mkt_pric_rate": [0.4, 1, 0.7],
    "fsthand_mkt_pric": 0.3,
    "scd_alu_price": [0.66, 1.98, 1.32],
    "glass_cullet_price": [0.01, 0.06, 0.035],
    "init_trust_boundaries": [-1, 1],
    "social_event_boundaries": [-1, 1]}

variable_params = {
    "consumers_network_type": ["small-world", "complete graph"],
    "init_manageEoL_rate": [0.1, 0.3, 0.6],
    "influencability": [0.2, 0.5, 0.8],
    "w_pbc": [0.2, 0.4, 0.8],
    "repairability": [0.25, 0.55],
    "recycling_learning_shape_factor": [-0.39, -0.8]}

# The variables parameters will be invoke along with the fixed parameters
# allowing for either or both to be honored.
batch_run = BatchRunner(
    ABM_CE_PV,
    variable_params,
    fixed_params,
    iterations=10,
    max_steps=51,
    model_reporters={
            "Year": lambda c: ABM_CE_PV.report_output(c, "year"),
            "Agents repairing": lambda c: ABM_CE_PV.count_EoL(c, "repairing"),
            "Agents selling": lambda c: ABM_CE_PV.count_EoL(c, "selling"),
            "Agents recycling": lambda c: ABM_CE_PV.count_EoL(c, "recycling"),
            "Agents landfilling": lambda c:
            ABM_CE_PV.count_EoL(c, "landfilling"),
            "Agents storing": lambda c: ABM_CE_PV.count_EoL(c, "hoarding"),
            "In operation": lambda c:
            ABM_CE_PV.report_output(c, "product_stock"),
            "End-of-life - repaired": lambda c:
            ABM_CE_PV.report_output(c, "product_repaired"),
            "End-of-life - sold": lambda c:
            ABM_CE_PV.report_output(c, "product_sold"),
            "End-of-life - recycled": lambda c:
            ABM_CE_PV.report_output(c, "product_recycled"),
            "End-of-life - landfilled": lambda c:
            ABM_CE_PV.report_output(c, "product_landfilled"),
            "End-of-life - stored": lambda c:
            ABM_CE_PV.report_output(c, "product_hoarded"),
            "Average landfilling cost": lambda c:
            ABM_CE_PV.report_output(c, "average_landfill_cost"),
            "Average storing cost": lambda c:
            ABM_CE_PV.report_output(c, "average_hoarding_cost"),
            "Average recycling cost": lambda c:
            ABM_CE_PV.report_output(c, "average_recycling_cost"),
            "Average repairing cost": lambda c:
            ABM_CE_PV.report_output(c, "average_repairing_cost"),
            "Average selling cost": lambda c:
            ABM_CE_PV.report_output(c, "average_second_hand_price")})

batch_run.run_all()

run_data = batch_run.get_model_vars_dataframe()
run_data.to_csv("BatchRun.csv")

