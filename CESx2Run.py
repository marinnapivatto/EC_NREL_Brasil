# -*- coding:utf-8 -*-
"""
Created on Wed Nov 21 12:43 2019

@author Julien Walzberg - Julien.Walzberg@nrel.gov

Run
"""

from ABM_CE_PV_Model import *
import matplotlib.pyplot as plt


model = ABM_CE_PV()

# Run model several time
for j in range(1):
    # Reinitialize model
    model.__init__()
    i = 0
    for i in range(51):
        model.step()

    # Get results in a pandas DataFrame
    results_model = model.datacollector.get_model_vars_dataframe()
    results_agents = model.datacollector.get_agent_vars_dataframe()
    results_model.to_csv("Results_model_run%s.csv" % j)
    results_agents.to_csv("Results_agents.csv")

# Draw the networks of agents and results
# nx.draw(model.H1, node_color="lightskyblue")
# nx.draw(model.H2, node_color="purple")
# nx.draw(model.H3, node_color="chocolate", edge_color="white")
# nx.draw(model.G, with_labels=False)
results_model[results_model.columns[1:6]].plot()
results_model[results_model.columns[7:12]].plot()
plt.show()  # draw graph as desired and plot outputs
