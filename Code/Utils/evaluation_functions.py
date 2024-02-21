from sklearn.metrics import confusion_matrix,classification_report,precision_recall_fscore_support,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluation_metrics(y_true,y_hat):
    accuracy = accuracy_score(y_true,y_hat)
    precision,recall,fscore,support = precision_recall_fscore_support(y_true,y_hat,average='weighted')
    return accuracy,precision,recall,fscore,support


def report_statistics(y_true,y_hat):
    report = classification_report(y_true,y_hat,target_names=['No topic shift','Topic shift'],zero_division=0)
    print(report)
    return report

def plot_confusion(y_true,y_hat):
    confusion = confusion_matrix(y_true,y_hat)
    sns.heatmap(confusion, annot=True,cmap='Blues',fmt='g')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.show()