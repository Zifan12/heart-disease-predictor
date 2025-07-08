from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc

def print_metrics(y_true, y_pred):
    """
    Print and return classification performance metrics.
    
    Args:
        y_true (array): True target labels
        y_pred (array): Predicted labels

    Returns:
        dict: Dictionary that contains accuracy, F1 score, precision, and recall.
    """
    acc = accuracy_score(y_true, y_pred)

    # Generate full classification report as a dictionary

    class_report = classification_report(y_true, y_pred, output_dict=True)
    # Extract weighted average metrics from the report
    f1 = class_report['weighted avg']['f1-score']
    precision = class_report['weighted avg']['precision']
    recall = class_report['weighted avg']['recall']

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

def plot_confusion_matrix(y_true, y_pred, model_name, return_fig=False):
    """
    Generate and save or return a confusion matrix.

    Args:
        y_true (array): True target labels
        y_pred (array): Prediction labels
        model_name (str): Name of the model 
        return_fig (bool): If True, return the matplotlib figure instead of saving it.

    Returns:
        Figure or None: igure object if return_fig is True, otherwise None.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix {model_name}')

    fig.tight_layout()
    
    if return_fig == True:
        return fig
    else:
        filename = f"plots/{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        fig.savefig(filename)
        plt.close()

def cross_validate_model(model, X, y, cv=5):
    """
    Perform k-fold cross-validation and print accuracy scores.

    Args: 
        model (estimator): Scikit-learn estimator.
        X (array): Feature data
        Y (array): Target labels
        cv (int): Number of cross-validation folds
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-Validation scores: {scores}")
    print(f"Mean CV accuracy: {scores.mean():.4f}")
    return scores

def plot_roc_curve(y_true, y_probability, model):
    """
    Plot and safve the ROC curve for a classifier

    Args:
        y_true (array): True target labels
        y_probability (array): Predicted probabilities for class 1 (has heart disease)
    
    Returns:
        None
    """

    fpr, tpr, _ = roc_curve(y_true, y_probability) 
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve For {model}')
    ax.legend(loc='lower right')
    ax.grid(True)
      
    filename = f"plots/{model.lower().replace(' ', '_')}_roc_curve.png"
    fig.savefig(filename)
    plt.close(fig)

