# -*- coding:utf-8 -*-
"""
Created on Wed Nov 21 12:43 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Run
"""
# TO DO: continue implementing run
# start year by year, 2019 to 2070

from CESx2ToolModel_v3 import *
import matplotlib.pyplot as plt
import numpy as np

model = CESx2()

# Run model
for i in range(51):
    model.step()
    model.clock = model.clock + 1

# Get results in a pandas DataFrame
results_model = model.datacollector.get_model_vars_dataframe()
results_agents = model.datacollector.get_agent_vars_dataframe()
results_model.to_csv("Results_model.csv")
results_agents.to_csv("Results_agents.csv")


# Draw the network of agents and results
def color_agents(step, column):
    color_map = []
    for node in model.G:
        agents_df = results_agents.loc[step, column]
        if agents_df[node]:
            color_map.append('green')
        else:
            color_map.append('blue')
    return color_map


nx.draw(model.G, node_color=color_agents(51, "Recycling"), with_labels=True)
results_model[results_model.columns[0:2]].plot()
results_model[results_model.columns[2:5]].plot()
plt.show()
