import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from w_optim import WeightNet

def train_optimizer(catboost_clf, lightgbm_clf, random_forest_clf, X_test, y_test, num_epochs=1500):
    test_predictions = np.array([
        catboost_clf.predict_proba(X_test)[:, 1],
        lightgbm_clf.predict_proba(X_test)[:, 1],
        random_forest_clf.predict_proba(X_test)[:, 1]
    ])

    catboost_clf_proba = catboost_clf.predict_proba(X_test)
    lightgbm_clf_proba = lightgbm_clf.predict_proba(X_test)
    random_forest_clf_proba = random_forest_clf.predict_proba(X_test)

    y_true = y_test
    input_tensor = torch.tensor(test_predictions.T, dtype=torch.float32)
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)

    model = WeightNet(3, 3)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = num_epochs
    best_f1 = 0
    best_weights = None
    f1_scores = []

    # Smoothed f1_score prevents the final metric from falling
    smoothed_f1_scores = []
    alpha = 0.01

    for epoch in range(num_epochs):
        # Nullify gradients
        optimizer.zero_grad()

        # Run the data through the neural network and get predictions
        weight_preds = model(input_tensor)

        # Average the predictions of the models, taking into account the predicted weights
        weighted_test_preds = torch.sum(weight_preds * input_tensor, axis=1)
    
        # Calculate the error based on the loss function
        loss = criterion(weighted_test_preds, y_true_tensor)
    
        # Calculating gradients
        loss.backward()

        # Updating Model Parameters
        optimizer.step()

        # Converting predictions back to numpy format
        weighted_test_preds_np = weighted_test_preds.detach().numpy()
        weight_preds_np = weight_preds.detach().numpy()

        # Calculate f1_score
        y_pred = (weighted_test_preds_np > 0.5).astype(int)
        current_f1 = f1_score(y_true, y_pred)
    
        # Adding f1_score to the list
        f1_scores.append(current_f1)
    
        # Calculate an exponential moving average
        if epoch == 0:
            smoothed_f1 = current_f1
            smoothed_f1_scores.append(current_f1)
        else:
            smoothed_f1 = alpha * current_f1 + (1 - alpha) * smoothed_f1_scores[-1]
            smoothed_f1_scores.append(smoothed_f1)
        
        # Save the best weight
        if smoothed_f1 > best_f1:
            best_f1 = smoothed_f1
            best_weights = weight_preds_np.mean(axis=0)
        
        # Final f1
        weighted_proba_final = np.average([
                    catboost_clf_proba, 
                    lightgbm_clf_proba, 
                    random_forest_clf_proba
        ], axis=0, weights=best_weights)

        bin_weighted_proba_final = (weighted_proba_final > 0.5).astype(int)[:, 1]

        final_f1 = f1_score(y_true, bin_weighted_proba_final)

    result = {}
    result['best_weights'] = best_weights
    result['final_f1'] = final_f1
    result['f1_scores'] = f1_scores
    result['smoothed_f1_scores'] = smoothed_f1_scores

    return result