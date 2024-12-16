#%% md
# # Result Summary
#%%
from
results = {
    "TF-IDF Model": {
        "MSE Sentence 1": a_mse1,
        "MSE Sentence 2": a_mse2,
        "Pearson Train": a_pc,
        "Pearson Test": a_pc_t
    },
    "Word2Vec Model": {
        "MSE Sentence 1": b_mse1,
        "MSE Sentence 2": b_mse2,
        "Pearson Train": b_pc,
        "Pearson Test": b_pc_t
    },
    "BioSentVec Model": {
        "MSE Sentence 1": c_mse1,
        "MSE Sentence 2": c_mse2,
        "Pearson Train": c_pc,
        "Pearson Test": c_pc_t
    },
    "CNN Model": {
        "MSE Dev": d_mse,
        "Best MSE Test": a_best_mse
    },
    "LSTM Model": {
        "Best MSE Test": best_mse
    }
}

#%%
def print_results(results):
    for model, metrics in results.items():
        print(f"{model}:")
        for metric, value in metrics.items():
            print(f"  {metric:<40}: {value}")
        print("-" * 60)

print_results(results)
