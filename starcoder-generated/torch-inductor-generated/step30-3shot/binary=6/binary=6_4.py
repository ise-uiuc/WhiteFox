
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x, y = x1.size()
        v1 = x1.view(x, -1)
        v2 = torch.ones([y, 1], dtype=torch.float32).cuda()
        v3 = torch.mv(v2, v1)
        return x2[v3]-x1
    
# Creating new, arbitrary tensor 
v4 = torch.rand([9, 4])

# Initializing the model
m = Model()
    
# Inputs to the model    
x1, x2 = v4, torch.randn(1, 1, 64, 64)
