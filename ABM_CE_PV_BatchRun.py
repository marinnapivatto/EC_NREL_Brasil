# -*- coding:utf-8 -*-
"""
Created on Wed Nov 21 12:43 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Run - batch of simulations with final state of outputs
"""

from ABM_CE_PV_Model import *
from mesa.batchrunner import BatchRunner
from SALib.sample import saltelli
from SALib.analyze import sobol
from copy import deepcopy
import time


# Batch run model
t0 = time.time()
all_fixed_params = {
    "seed": None,
    "num_consumers": 1000,
    "consumers_node_degree": 10,
    "consumers_network_type": "small-world",
    "rewiring_prob": 0.1,
    "num_recyclers": 16,
    "num_producers": 60,
    "prod_n_recyc_node_degree": 5,
    "prod_n_recyc_network_type": "small-world",
    "num_refurbishers": 15,
    "consumers_distribution": {
        "residential": 1, "commercial": 0., "utility": 0.},
    "init_eol_rate": {
        "repair": 0.005, "sell": 0.02, "recycle": 0.1, "landfill": 0.4375,
        "hoard": 0.4375},
    "init_purchase_choice": {"new": 0.98, "used": 0.02, "certified": 0},
    "total_number_product": [38, 38, 38, 38, 38, 38, 38, 139, 251,
                             378, 739, 1670, 2935, 4146, 5432, 6525,
                             3609, 4207, 4905, 5719],
    "product_distribution": {"residential": 1,
                             "commercial": 0., "utility": 0.},
    "product_growth": [0.166, 0.045],
    "growth_threshold": 10,
    "failure_rate_alpha": [2.4928, 5.3759, 3.93495],
    "hoarding_cost": [0, 0.001, 0.0005],
    "landfill_cost": [0.003, 0.009, 0.006],
    "theory_of_planned_behavior": {"residential": True, "commercial": True,
                                   "utility": True},
    "w_sn_eol": 0.27,
    "w_pbc_eol": 0.44,
    "w_a_eol": 0.39,
    "w_sn_reuse": 0.497,
    "w_pbc_reuse": 0.382,
    "w_a_reuse": 0.464,
    "product_lifetime": 30,
    "all_EoL_pathways": {"repair": True, "sell": True, "recycle": True,
                         "landfill": True, "hoard": True},
    "max_storage": [1, 8, 4],
    "att_distrib_param_eol": [0.5435, 0.1],
    "att_distrib_param_reuse": [0.35, 0.262],
    "original_recycling_cost": [0.106, 0.128, 0.117],
    "recycling_learning_shape_factor": -0.39,
    "repairability": 0.55,
    "original_repairing_cost": [0.1, 0.45, 0.28],
    "repairing_learning_shape_factor": -0.31,
    "scndhand_mkt_pric_rate": [0.4, 1, 0.7],
    "fsthand_mkt_pric": 0.3,
    "refurbisher_margin": [0.03, 0.45, 0.24],
    "purchase_choices": {"new": True, "used": True, "certified": False},
    "init_trust_boundaries": [-1, 1],
    "social_event_boundaries": [-1, 1],
    "social_influencability_boundaries": [0, 1],
    "trust_threshold": 0.5,
    "knowledge_threshold": 0.5,
    "willingness_threshold": 0.5,
    "self_confidence_boundaries": [0, 1],
    "product_mass_fractions": {
        "Product": 1, "Aluminum": 0.08, "Glass": 0.76, "Copper": 0.01,
        "Insulated cable": 0.012, "Silicon": 0.036, "Silver": 0.00032},
    "established_scd_mkt": {
        "Product": True, "Aluminum": True, "Glass": True, "Copper": True,
        "Insulated cable": True, "Silicon": False, "Silver": False},
    "scd_mat_prices": {
        "Product": [np.nan, np.nan, np.nan], "Aluminum": [0.66, 1.98, 1.32],
        "Glass": [0.01, 0.06, 0.035], "Copper": [3.77, 6.75, 5.75],
        "Insulated cable": [3.22, 3.44, 3.33], "Silicon": [2.20, 3.18, 2.69],
        "Silver": [453, 653, 582]},
    "virgin_mat_prices": {
        "Product": [np.nan, np.nan, np.nan], "Aluminum": [1.76, 2.51, 2.14],
        "Glass": [0.04, 0.07, 0.055], "Copper": [4.19, 7.50, 6.39],
        "Insulated cable": [3.22, 3.44, 3.33], "Silicon": [2.20, 3.18, 2.69],
        "Silver": [453, 653, 582]},
    "material_waste_ratio": {
        "Product": 0., "Aluminum": 0., "Glass": 0., "Copper": 0.,
        "Insulated cable": 0., "Silicon": 0.4, "Silver": 0.},
    "recovery_fractions": {
        "Product": np.nan, "Aluminum": 0.92, "Glass": 0.85, "Copper": 0.72,
        "Insulated cable": 1., "Silicon": 0., "Silver": 0.},
    "product_average_wght": 0.1,
    "mass_to_function_reg_coeff": 0.03,
    "recycling_states": ['Texas', 'Arizona', 'Oregon', 'Oklahoma',
                         'Wisconsin', 'Ohio', 'Kentucky', 'South Carolina'],
    "transportation_cost": 0.0314,
    "used_product_substitution_rate": [0.6, 1, 0.8],
    "imperfect_substitution": 0,
    "epr_business_model": False,
    "recycling_process": {"frelp": False, "asu": False, "hybrid": False},
    "industrial_symbiosis": False,
    "dynamic_lifetime_model": {
        "Dynamic lifetime": False, "d_lifetime_intercept": 15.9,
        "d_lifetime_reg_coeff": 0.87, "Seed": False, "Year": 5,
        "avg_lifetime": 50},
    "extended_tpb": {
        "Extended tpb": False, "w_convenience": 0.28, "w_knowledge": -0.51,
        "knowledge_distrib": [0.5, 0.49]},
    "seeding": {"Seeding": False, "Year": 10, "number_seed": 50},
    "seeding_recyc": {"Seeding": False, "Year": 10, "number_seed": 50,
                      "discount": 0.35}}

# The variables parameters will be invoke along with the fixed parameters
# allowing for either or both to be honored.
FullFactorial = False
if FullFactorial:
    for i in range(1):
        if i < len(range(1)):
            calibration_n_sensitivity = True
            if calibration_n_sensitivity:
                variable_params = {}
                if i == 0:
                    variable_params = {
                        "seed": list(range(30)), "calibration_n_sensitivity_3":
                            [1E-6, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                             0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                             0.9, 0.95, 1], "w_sn_eol": 0,
                        "w_pbc_eol": 0.44,
                        "w_a_eol": 0}
                if i == 1:
                    variable_params = {}
            else:
                variable_params = {"seed": list(range(1))}
            fixed_params = all_fixed_params.copy()
            for key in variable_params.keys():
                fixed_params.pop(key)
        else:
            # Enables running different types of sensitivity analysis
            fixed_params = {}
            variable_params = {}
        batch_run = BatchRunner(
            ABM_CE_PV,
            variable_params,
            fixed_params,
            iterations=1,
            max_steps=30,
            model_reporters={
                    "Year": lambda c: ABM_CE_PV.report_output(c, "year"),
                    "Agents repairing": lambda c:
                    ABM_CE_PV.count_EoL(c, "repairing"),
                    "Agents selling": lambda c:
                    ABM_CE_PV.count_EoL(c, "selling"),
                    "Agents recycling": lambda c:
                    ABM_CE_PV.count_EoL(c, "recycling"),
                    "Agents landfilling": lambda c:
                    ABM_CE_PV.count_EoL(c, "landfilling"),
                    "Agents storing": lambda c:
                    ABM_CE_PV.count_EoL(c, "hoarding"),
                    "Agents buying new": lambda c:
                    ABM_CE_PV.count_EoL(c, "buy_new"),
                    "Agents buying used": lambda c:
                    ABM_CE_PV.count_EoL(c, "buy_used"),
                    "Agents buying certified": lambda c:
                    ABM_CE_PV.count_EoL(c, "certified"),
                    "Total product": lambda c:
                    ABM_CE_PV.report_output(c, "product_stock"),
                    "New product": lambda c:
                    ABM_CE_PV.report_output(c, "product_stock_new"),
                    "Used product": lambda c:
                    ABM_CE_PV.report_output(c, "product_stock_used"),
                    "New product_mass": lambda c:
                    ABM_CE_PV.report_output(c, "prod_stock_new_mass"),
                    "Used product_mass": lambda c:
                    ABM_CE_PV.report_output(c, "prod_stock_used_mass"),
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
                    "eol - new repaired weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_new_repaired"),
                    "eol - new sold weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_new_sold"),
                    "eol - new recycled weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_new_recycled"),
                    "eol - new landfilled weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_new_landfilled"),
                    "eol - new stored weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_new_hoarded"),
                    "eol - used repaired weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_used_repaired"),
                    "eol - used sold weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_used_sold"),
                    "eol - used recycled weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_used_recycled"),
                    "eol - used landfilled weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_used_landfilled"),
                    "eol - used stored weight": lambda c:
                    ABM_CE_PV.report_output(c, "product_used_hoarded"),
                    "Average landfilling cost": lambda c:
                    ABM_CE_PV.report_output(c, "average_landfill_cost"),
                    "Average storing cost": lambda c:
                    ABM_CE_PV.report_output(c, "average_hoarding_cost"),
                    "Average recycling cost": lambda c:
                    ABM_CE_PV.report_output(c, "average_recycling_cost"),
                    "Average repairing cost": lambda c:
                    ABM_CE_PV.report_output(c, "average_repairing_cost"),
                    "Average selling cost": lambda c:
                    ABM_CE_PV.report_output(c, "average_second_hand_price"),
                    "Recycled material volume": lambda c:
                    ABM_CE_PV.report_output(c, "recycled_mat_volume"),
                    "Recycled material value": lambda c:
                    ABM_CE_PV.report_output(c, "recycled_mat_value"),
                    "Producer costs": lambda c:
                    ABM_CE_PV.report_output(c, "producer_costs"),
                    "Consumer costs": lambda c:
                    ABM_CE_PV.report_output(c, "consumer_costs"),
                    "Recycler costs": lambda c:
                    ABM_CE_PV.report_output(c, "recycler_costs"),
                    "Refurbisher costs": lambda c:
                    ABM_CE_PV.report_output(c, "refurbisher_costs")})
        batch_run.run_all()
        run_data = batch_run.get_model_vars_dataframe()
        run_data.to_csv("BatchRun%s.csv" % i)
else:
    list_variables = \
        ["all_EoL_pathways", "recovery_fractions", "num_recyclers",
         "original_recycling_cost", "landfill_cost", "original_repairing_cost",
         "scndhand_mkt_pric_rate", "max_storage", "att_distrib_param_eol",
         "att_distrib_param_reuse", "num_consumers", "product_lifetime",
         "recycling_learning_shape_factor", "rewiring_prob", "w_sn_eol",
         "w_a_eol", "w_pbc_eol", "w_sn_reuse", "w_a_reuse", "w_pbc_reuse",
         "transportation_cost", "repairability"]
    problem = {
        'num_vars': 22,
        'names': ["all_EoL_pathways", "recovery_fractions", "num_recyclers",
                  "original_recycling_cost", "landfill_cost",
                  "original_repairing_cost", "scndhand_mkt_pric_rate",
                  "max_storage", "att_distrib_param_eol",
                  "att_distrib_param_reuse", "num_consumers",
                  "product_lifetime", "recycling_learning_shape_factor",
                  "rewiring_prob", "w_sn_eol", "w_a_eol", "w_pbc_eol",
                  "w_sn_reuse", "w_a_reuse", "w_pbc_reuse",
                  "transportation_cost", "repairability"],
        'bounds': [[1E-6, 1], [1E-6, 1], [16, 96], [1E-6, 1], [1E-6, 2],
                   [1E-6, 1], [1E-6, 1], [1E-6, 2], [1E-6, 1], [1E-6, 1],
                   [200, 1000], [10, 60], [1E-6, 0.6],
                   [1E-6, 1], [1E-6, 1], [1E-6, 1], [1E-6, 1], [1E-6, 1],
                   [1E-6, 1], [1E-6, 1], [1E-6, 0.1], [1E-6, 0.999999]]}
    X = saltelli.sample(problem, 20)
    baseline_row = np.array([1E-6, 1E-6, 16.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5435,
                             0.35, 1000.0, 30.0, 0.39, 0.1, 0.27, 0.39, 0.44,
                             0.497, 0.464, 0.382, 0.0314, 0.55])
    X = np.vstack((X, baseline_row))
    for x in range(X.shape[1]):
        lower_bound = deepcopy(baseline_row)
        bounds = problem['bounds'][x]
        if lower_bound[x] != bounds[0]:
            lower_bound[x] = bounds[0]
            X = np.vstack((X, lower_bound))
        upper_bound = deepcopy(baseline_row)
        if upper_bound[x] != bounds[1]:
            upper_bound[x] = bounds[1]
            X = np.vstack((X, upper_bound))
    appended_data = []
    for i in range(X.shape[0]):
        print("Sobol matrix line: ", i, " out of ", X.shape[0])
        fixed_params = deepcopy(all_fixed_params)
        for j in range(X.shape[1]):
            value_to_change = X[i][j]
            variable_to_change = list_variables[j]
            if j < 1:
                if value_to_change > 0.5:
                    fixed_params[variable_to_change]["landfill"] = False
                else:
                    fixed_params[variable_to_change]["landfill"] = True
            elif j < 2:
                print(value_to_change, fixed_params[variable_to_change])
                for key in fixed_params[variable_to_change].keys():
                    fixed_params[variable_to_change][key] += (
                            1 - fixed_params[variable_to_change][key]) * \
                                                             value_to_change
                print(fixed_params[variable_to_change])
            elif j < 3:
                fixed_params[variable_to_change] = int(value_to_change)
                model_instance = ABM_CE_PV()
                num_states_to_add = int(value_to_change / 2) - \
                    len(fixed_params['recycling_states'])
                for count_new_states in range(num_states_to_add):
                    choice_states = [
                        x for x in model_instance.all_states
                        if x not in fixed_params['recycling_states']]
                    fixed_params['recycling_states'].append(
                        random.choice(choice_states))
            elif j < 8:
                fixed_params[variable_to_change] = [
                    x * value_to_change for x in
                    fixed_params[variable_to_change]]
            elif j < 10:
                fixed_params[variable_to_change][0] = value_to_change
            elif j < 12:
                fixed_params[variable_to_change] = int(value_to_change)
            elif j < 13:
                fixed_params[variable_to_change] = -1 * value_to_change
            else:
                fixed_params[variable_to_change] = value_to_change
        variable_params = {"seed": list(range(1))}
        fixed_params.pop("seed")
        batch_run = BatchRunner(
            ABM_CE_PV,
            variable_params,
            fixed_params,
            iterations=1,
            max_steps=1,
            model_reporters={
                "Year": lambda c: ABM_CE_PV.report_output(c, "year"),
                "Agents repairing": lambda c:
                ABM_CE_PV.count_EoL(c, "repairing"),
                "Agents selling": lambda c:
                ABM_CE_PV.count_EoL(c, "selling"),
                "Agents recycling": lambda c:
                ABM_CE_PV.count_EoL(c, "recycling"),
                "Agents landfilling": lambda c:
                ABM_CE_PV.count_EoL(c, "landfilling"),
                "Agents storing": lambda c:
                ABM_CE_PV.count_EoL(c, "hoarding"),
                "Agents buying new": lambda c:
                ABM_CE_PV.count_EoL(c, "buy_new"),
                "Agents buying used": lambda c:
                ABM_CE_PV.count_EoL(c, "buy_used"),
                "Agents buying certified": lambda c:
                ABM_CE_PV.count_EoL(c, "certified"),
                "Total product": lambda c:
                ABM_CE_PV.report_output(c, "product_stock"),
                "New product": lambda c:
                ABM_CE_PV.report_output(c, "product_stock_new"),
                "Used product": lambda c:
                ABM_CE_PV.report_output(c, "product_stock_used"),
                "New product_mass": lambda c:
                ABM_CE_PV.report_output(c, "prod_stock_new_mass"),
                "Used product_mass": lambda c:
                ABM_CE_PV.report_output(c, "prod_stock_used_mass"),
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
                "eol - new repaired weight": lambda c:
                ABM_CE_PV.report_output(c, "product_new_repaired"),
                "eol - new sold weight": lambda c:
                ABM_CE_PV.report_output(c, "product_new_sold"),
                "eol - new recycled weight": lambda c:
                ABM_CE_PV.report_output(c, "product_new_recycled"),
                "eol - new landfilled weight": lambda c:
                ABM_CE_PV.report_output(c, "product_new_landfilled"),
                "eol - new stored weight": lambda c:
                ABM_CE_PV.report_output(c, "product_new_hoarded"),
                "eol - used repaired weight": lambda c:
                ABM_CE_PV.report_output(c, "product_used_repaired"),
                "eol - used sold weight": lambda c:
                ABM_CE_PV.report_output(c, "product_used_sold"),
                "eol - used recycled weight": lambda c:
                ABM_CE_PV.report_output(c, "product_used_recycled"),
                "eol - used landfilled weight": lambda c:
                ABM_CE_PV.report_output(c, "product_used_landfilled"),
                "eol - used stored weight": lambda c:
                ABM_CE_PV.report_output(c, "product_used_hoarded"),
                "Average landfilling cost": lambda c:
                ABM_CE_PV.report_output(c, "average_landfill_cost"),
                "Average storing cost": lambda c:
                ABM_CE_PV.report_output(c, "average_hoarding_cost"),
                "Average recycling cost": lambda c:
                ABM_CE_PV.report_output(c, "average_recycling_cost"),
                "Average repairing cost": lambda c:
                ABM_CE_PV.report_output(c, "average_repairing_cost"),
                "Average selling cost": lambda c:
                ABM_CE_PV.report_output(c,
                                        "average_second_hand_price"),
                "Recycled material volume": lambda c:
                ABM_CE_PV.report_output(c, "recycled_mat_volume"),
                "Recycled material value": lambda c:
                ABM_CE_PV.report_output(c, "recycled_mat_value"),
                "Producer costs": lambda c:
                ABM_CE_PV.report_output(c, "producer_costs"),
                "Consumer costs": lambda c:
                ABM_CE_PV.report_output(c, "consumer_costs"),
                "Recycler costs": lambda c:
                ABM_CE_PV.report_output(c, "recycler_costs"),
                "Refurbisher costs": lambda c:
                ABM_CE_PV.report_output(c, "refurbisher_costs"),
                "Refurbisher costs w margins": lambda c:
                ABM_CE_PV.report_output(c, "refurbisher_costs_w_margins")})
        batch_run.run_all()
        run_data = batch_run.get_model_vars_dataframe()
        for k in range(X.shape[1]):
            run_data["x_%s" % k] = X[i][k]
        appended_data.append(run_data)
    appended_data = pd.concat(appended_data)
    appended_data["Y1"] = \
        (appended_data["End-of-life - recycled"]) / \
        (appended_data["End-of-life - recycled"] +
         appended_data["End-of-life - repaired"] +
         appended_data["End-of-life - sold"] +
         appended_data["End-of-life - landfilled"] +
         appended_data["End-of-life - stored"])
    appended_data["Y2"] = \
        (appended_data["End-of-life - repaired"] +
         appended_data["End-of-life - sold"]) / \
        (appended_data["End-of-life - recycled"] +
         appended_data["End-of-life - repaired"] +
         appended_data["End-of-life - sold"] +
         appended_data["End-of-life - landfilled"] +
         appended_data["End-of-life - stored"])
    params_usage = deepcopy(all_fixed_params)
    params_usage["product_mass_fractions"].pop("Product")
    params_usage["recovery_fractions"].pop("Product")
    params_usage["scd_mat_prices"].pop("Product")
    appended_data["Y3"] = \
        (appended_data["eol - new recycled weight"] +
         appended_data["eol - used recycled weight"]) * \
        sum(params_usage["product_mass_fractions"][k] *
            params_usage["recovery_fractions"][k] for k in
            params_usage["product_mass_fractions"])
    appended_data["Y4"] = \
        (appended_data["eol - new recycled weight"] +
         appended_data["eol - used recycled weight"]) * \
        sum(params_usage["product_mass_fractions"][k] *
            params_usage["recovery_fractions"][k] *
            params_usage["scd_mat_prices"][k][2] for k in
            params_usage["product_mass_fractions"])
    # We do not include consumer costs to avoid double counting but
    # we can present them separately
    appended_data["Y5"] = \
        (appended_data["Recycler costs"] +
         appended_data["Refurbisher costs"] +
         appended_data["Producer costs"])
    appended_data["Y6"] = appended_data["Used product"] / \
        appended_data["New product"]
    appended_data.to_csv("SobolBatchRun.csv")
    data_out = appended_data.filter(["seed", "x_0", "x_1", "x_2", "x_3", "x_4",
                                     "x_5", "x_6", "x_7", "x_8", "x_9", "x_10",
                                     "x_11", "x_12", "x_13", "x_14", "x_15",
                                     "x_16", "x_17", "x_18", "x_19", "x_20",
                                     "x_21", "x_22", "Y1", "Y2", "Y3", "Y4",
                                     "Y5", "Y6"], axis=1)
    data_out.to_csv("DataML.csv")
t1 = time.time()
print(t1 - t0)
print("Done!")
