import matplotlib.pyplot as plt

class VisualData:
    def showUsing_Pyplot(X,Y):

        xpoints = array(X)
        ypoints = array(Y)
        
        plt.plot(xpoints, ypoints, marker = "", linestyle = "dashed", )
        #marker = this attribute is used for mark the plot 
        
        # To add label:
        plt.xlabel("Probability of people have same Birthday")
        plt.ylabel("No. of people")
        
        #To add grid:
        plt.grid()

        #To show
        plt.show()

    def scatterVis(self, X, Y):
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        plt.title("Non Linear / Linear Data")
        plt.show()

# df= VisualData()
# df.scatterVis(10,10)