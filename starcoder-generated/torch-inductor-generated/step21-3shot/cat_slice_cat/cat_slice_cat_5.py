
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t1, t2):
        t3 = torch.cat([t1, t2], dim=1)
        t4 = t3[:,0:-10]
        t5 = torch.cat([t3,t4],1)
        return t5
 
# Initializing the model
m = Model()
 
# Input to the model 
x1 = torch.randn(1, 10) 
x2 = torch.randn(1, 10)
