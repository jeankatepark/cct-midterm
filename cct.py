'''
cct midterm
'''

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

#Load the Data

url = "https://raw.githubusercontent.com/joachimvandekerckhove/cogs107s25/refs/heads/main/1-mpt/data/plant_knowledge.csv"

def load_plant_data(url):
    df = pd.read_csv(url)
    data = df.drop(columns=['Informant']).values
    return data

data = load_plant_data(url)  
N, M = data.shape  # N: informants, M: items

#Implement the Model
with pm.Model() as cct_model:
    # Define Priors
    D = pm.Uniform("D", 0, 1, shape=N)  
    Z = pm.Bernoulli("Z", p=0.5, shape=M)      

    # Reshape 
    D_reshaped = D[:, None]  

    # Probability of each X_ij
    p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)

    # Likelihood
    X = pm.Bernoulli("X", p=p, observed=data)

    # Perform Inference
    trace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.9, return_inferencedata=True)

# Analyze Results
summary = az.summary(trace, var_names=["D", "Z"])
summary.to_csv("summary.csv")
print(summary)

# Visualize posterior distributions for D and Z
az.plot_posterior(trace, var_names=["D"])
plt.tight_layout()
plt.savefig("posterior_D.png")

az.plot_posterior(trace, var_names=["Z"])
plt.tight_layout()
plt.savefig("posterior_Z.png")

# Compare with Naive ggregation
naive_consensus = np.round(data.mean(axis=0)).astype(int)
posterior_mean_Z = trace.posterior['Z'].mean(dim=["chain", "draw"]).values.round().astype(int)

print("\nNaive vs. Bayesian Consensus Answers:")
for i, (naive, bayes) in enumerate(zip(naive_consensus, posterior_mean_Z)):
    print(f"Q{i+1}: Naive={naive}, CCT={bayes}")


