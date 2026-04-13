import matplotlib.pyplot as plt

def plot_results(y_true, y_pred, path):

    plt.figure(figsize=(10,5))
    plt.plot(y_true.values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.savefig(path)
    plt.close()