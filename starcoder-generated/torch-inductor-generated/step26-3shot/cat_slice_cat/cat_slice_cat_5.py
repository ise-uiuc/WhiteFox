
class Model(torch.nn.Module):
     def __init__(self):
         super().__init__()
 
     def forward(self, x):
         v1 = torch.cat([x[0], x[1]])
         v2 = torch.cat(v1, dim=1)
         return v2
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 3024413)
x2 = torch.randn(1, 2626332)
x3 = torch.randn(1, 2335121)
