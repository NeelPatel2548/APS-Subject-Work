
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

class importingFiles:

    def __init__(self):
        self.data = load_breast_cancer()
    
        self.x = pd.DataFrame(
            self.data.data,
            columns = self.data.feature_names
        )
    
        self.y = pd.Series((self.data.target == 0).astype(int), name = "malignant")
        self.class_counts = self.y.value_counts().sort_index()
        self.class_distrbtion =  pd.DataFrame({
                    "Class": self.data.target_names,
                    "Count": self.class_counts.values,
                    "Probability": self.class_counts.values / len(self.y)
                })

    def loadData(self):
        print(self.y.value_counts())

        print("Feature matrix shape: ", self.x.shape)
        print("Target shape: ", self.y.shape)
        print("Class names: ", self.data.target_names)


    def plottingdata(self):
        #ploting data using label

        self.class_distrbtion.plot(
            x = "Class",
            y = "Count",
            kind = "pie",
            legend= False,
            color = ["green", "black"]
        )

        plt.ylabel("Number of observation")
        plt.title("Class Distribution")
        plt.xticks(rotation = 0)
        plt.show()



imp = importingFiles()

imp.loadData()
# imp.plottingdata()