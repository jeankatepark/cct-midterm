'''
CCT Midterm
'''

#Set Up
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import sys


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
    D = pm.Beta("D", 2, 2, shape=N)  
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

# --- Constants ---

FIG_DIR = Path("figures") # Directory to save output figures

# --- Main execution block ---
if __name__ == "__main__":

    # --- Create Figure Directory ---
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Figures will be saved to: {FIG_DIR.resolve()}")
    except OSError as e:
        print(f"Error creating figure directory {FIG_DIR}: {e}", file=sys.stderr)

#Pair Plot 
az.plot_pair(
    trace,
    var_names=["D"],
    kind="kde",
    marginals = True
    )
plt.suptitle("CCT Model Pair Plot", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'cct_model_pairplot.png')

#Print Summary 
summary = az.summary(trace)
summary.to_csv("summary.csv")
print(summary)

#Did the Model Converge?
'''
No, the model did not completely converge, 
since the R-hat values are not close enough to 1, 
and the pair plot shows figures that are not unimodal. 
'''
print("\nDid the model converge?: No")

#Estimate Informant Competence
print("\nPosterior Mean Competence for Informants 1-10")
posterior_D = trace.posterior["D"].mean(dim=("chain", "draw")).values
for i, d in enumerate(posterior_D):
    print(f"Informant {i + 1} Competence = {d:.3f}")

#Visualize Posterior Distributions for Competence 
az.plot_posterior(trace, var_names=["D"])
plt.tight_layout()
plt.savefig(FIG_DIR / 'posterior_D.png')

#Identify Most and Least Competent Informants
most = np.argmax(posterior_D)
least = np.argmin(posterior_D)
print(f"Most competent informant: {most + 1} (D = {posterior_D[most]:.3f})")
print(f"Least competent informant: {least + 1} (D = {posterior_D[least]:.3f})")


#Estimate Consensus Answers 
print("\nPosterior Mean Probability for Consensus Answers")
posterior_Z = trace.posterior["Z"].mean(dim=("chain", "draw")).values
for j, z in enumerate(posterior_Z):
    print(f"Q{j + 1}: {z:.3f}")

#Determine Most Likely Consensus Answer Key 
consensus_answers = np.round(posterior_Z).astype(int)
print("\nMost Likely Consensus Answer Key")
for j, answer in enumerate(consensus_answers):
    print(f"Q{j + 1}: {answer}")

#Visualize Posterior Probabilities for Z 
az.plot_posterior(trace, var_names=["Z"])
plt.tight_layout()
plt.savefig(FIG_DIR / 'posterior_Z.png')


# Compare with Naive Aggregation
naive_consensus = np.round(data.mean(axis=0)).astype(int)
posterior_mean_Z = trace.posterior['Z'].mean(dim=["chain", "draw"]).values.round().astype(int)

print("\nNaive vs. Consensus Answer Key:")
for i, (naive, bayes) in enumerate(zip(naive_consensus, posterior_mean_Z)):
    print(f"Q{i+1}: Naive={naive}, CCT={bayes}")
'''
These differences may occur due to variation in informant competency.
'''

#Report
'''
My Cultural Consensus Theory (CCT) model is structured using PyMC, with the variable D representing the competence of each informant, 
and the variable Z representing the consensus for each question. For D's prior, I chose a Beta(2,2) distribution to represent how most 
informants would have average competence. For Z's prior, I chose a Bernoulli(0.5) distribution to represent equal probability for either 
of the binary answers (0 or 1). 

My results show that informants had competence means around 0.4-0.6 and mean probability for consensus answers were around 0.3-0.7. 
The model did not completely converge, since the R-hat values are not close enough to 1, and the pair plot shows not all figures are 
unimodal. There were several questions where the simple majority vote answer differed from the consensus answer key estimated by the 
CCT model, indicating how the CCT model takes the variability of informant competency into account. 
'''
