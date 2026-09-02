import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testing import testingData



from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class BCancer:
    def __init__(self):
        self.tes = testingData()
        self.y_test = self.tes.y_test
        self.actual_mali = (self.tes.y_test.values == 0).astype(int)
        self.y_pred = None    

    def makeConfusionMatrix(self):
        #construct a confusion matrix

        for tesh in self.tes.threshold:
            pred_mal = (
                self.tes.proablities[:,0] >= tesh
            ).astype(int)

            cm = confusion_matrix(
                self.actual_mali, 
                pred_mal
            )
            self.y_pred = pred_mal
            print(f"\nThreshold = {tesh}")
            print(cm)


        # TP - Mal - Mal
        # FP - Ben - Mal
        # FN - Mal - Ben 
        # TN - Ben - Ben

    def evaluationTable(self):
        result = []
        y_prob = self.tes.model.predict_proba(self.tes.x_test)[:, 1]
        for thresholld in [0.1,0.30, 0.50, 0.70, 0.9]:
            y_pred = (y_prob >= thresholld).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.tes.y_test, self.tes.y_pred).ravel()

            accuracy = accuracy_score(self.tes.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            result.append({
                "Threshold": thresholld,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

        result_table = pd.DataFrame(result)

        print(result_table)


bca = BCancer();

bca.makeConfusionMatrix()
bca.evaluationTable()