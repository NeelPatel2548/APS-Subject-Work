from GenerateData import GenerateData
from VisiualData import VisualData

class start_calling:

    def start_call(self):
        gen = GenerateData()
        X, Y = gen.linear_data()
        A, B = gen.non_linear_data()
        
        vis = VisualData()
        # vis.showUsing_Pyplot(rtr)
        vis.scatterVis(X, Y)
        vis.scatterVis(A,B)



start = start_calling()

start.start_call()