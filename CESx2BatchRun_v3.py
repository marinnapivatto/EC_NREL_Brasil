# -*- coding:utf-8 -*-
"""
Created on Wed Nov 21 12:43 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Run
"""
# TO DO: continue implementing run
# start year by year, 2019 to 2070
# find a way to get step by step values with batch run

from CESx2ToolModel_v3 import *
from mesa.batchrunner import BatchRunner

#model = CESx2()

# Batch run model
fixed_params = {
    "num_nodes": 100,
    "avg_node_degree": 5,
    "init_recycling_rate": 0.3,
    "total_number_product": [19, 19, 19, 19, 19, 19, 19, 58, 77, 157,
                                       246, 305, 496, 799, 1268, 2171, 2639,
                                       2227, 2357, 2488],
    "product_growth": [0.025, 0.09],
    "failure_rate_alpha": 5.4,
    "trust_increment": 0,
    "product_lifetime": 30}

variable_params = {
    "network_type": ["small-world", "complete graph"],
    "difficulty_recycling": [0, 0.25, 0.5, 0.75, 1],
    "independence_level": [0, 0.25, 0.5, 0.75, 1]}

# The variables parameters will be invoke along with the fixed parameters
# allowing for either or both to be honored.
batch_run = BatchRunner(
    CESx2,
    variable_params,
    fixed_params,
    iterations=10,
    max_steps=51,
    model_reporters={
            "Recycling_count": lambda c: CESx2.count_EoL(c, "recycling"),
            "Landfilling_count": lambda c: CESx2.count_EoL(c, "landfilling"),
            "Product_stock_count": lambda c:
            CESx2.count_stocks(c, "product_stock"),
            "Product_recycled_count": lambda c:
            CESx2.count_stocks(c, "product_recycled"),
            "Product_landfilled_count":
                lambda c: CESx2.count_stocks(c, "product_landfilled")})

batch_run.run_all()

run_data = batch_run.get_model_vars_dataframe()
run_data.to_csv("Run_data5.csv")

