from importingFiles import importingFiles

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

class testingData:

    def __init__(self):
        imp = importingFiles()
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(
            imp.x,
            imp.y,
            test_size=0.20,
            stratify=imp.y
        )
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        )

        self.model.fit(self.x_train,self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        self.proablities = self.model.predict_proba(self.x_test)

        self.results = pd.DataFrame({
            "Actual class": self.y_test.values,
            "P_malignant": self.proablities[:, 0],
            "P_Benign": self.proablities[:, 1]
        })

    def displayingTrainig(self):

        print("Training Size:", len(self.y_train))
        print("Testing Size:", len(self.y_test))

        print("\nTraining Propostions: ")
        print(self.y_train.value_counts(normalize = True).sort_index())

        print("\nTesting Propostions: ")
        print(self.y_test.value_counts(normalize = True).sort_index())

        # Results
        print("Predicted values:", self.y_pred)
        print("Actual values:", self.y_test.values)

        print("Probilities: ",self.proablities[:5])

        #printing result
        print(self.results.head())
        print(self.results.columns)

    def threshHoldTesting(self):
        results["Actual Labels"] = results["Actual class"].map({
    0: "P_malignant",
    1: "P_Benign"
})

print(
    results[
        ["Actual Labels", "P_malignant", "P_Benign"]
    ].head(10)
)

threshold = 0.50

results["Predicted_mal"] = (
    results["P_malignant"] >= threshold
).astype(int)


results["Predicted Labels"] = results[
    "Predicted_mal"
].map({
    1: "P_malignant",
    0: "P_Benign"
})

print(
    results[
        [
            "Actual Labels",
            "P_malignant",
            "Predicted Labels"

        ]
    ]
)



for threshold in [0.30, 0.50, 0.70]:
    predictions = (
        results["P_malignant"] >= threshold
    ).astype(int)

    print(
        f"Threshold = (threshold): ""LabWork 1108.ipynb"
        f"Predicted malignant cases: = {predictions.sum()}" 
    )



    ]
    


tds = testingData()

tds.displayingTrainig()