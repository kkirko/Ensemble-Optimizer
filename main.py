from data_generation import generate_data 
from train_models import train_models
from train_w_optim import train_optimizer
import matplotlib.pyplot as plt 

# Generate data
X_train, X_test, y_train, y_test = generate_data()

# Classifiers training
catboost_clf, lightgbm_clf, random_forest_clf = train_models(X_train, y_train)

# Weights optimizing
weights_optimizer = train_optimizer(catboost_clf, lightgbm_clf, random_forest_clf, X_test, y_test)
best_weights, final_f1, f1_scores, smoothed_f1_scores = weights_optimizer['best_weights'], weights_optimizer['final_f1'], weights_optimizer['f1_scores'], weights_optimizer['smoothed_f1_scores']

# Results
print(f"Best Weights: {best_weights}")
print(f"Best F1: {final_f1}")
print()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(f1_scores, label="Plain_f1_score")
plt.plot(smoothed_f1_scores, label="Smoothed f1_score")
plt.xlabel('Epochs')
plt.ylabel('f1_score')
plt.title('f1_score optimization')
plt.legend()
plt.show()



          
          

