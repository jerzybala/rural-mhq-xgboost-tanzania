import pickle
import pandas as pd

with open('one_hot_encoding_results.pkl', 'rb') as f:
    results = pickle.load(f)

print("Keys in saved results:")
for key in results.keys():
    print(f"  - {key}: {type(results[key])}")

print("\naggregated_importance content:")
print(results['aggregated_importance'])

print("\naggregated_shap content:")
print(results['aggregated_shap'])
