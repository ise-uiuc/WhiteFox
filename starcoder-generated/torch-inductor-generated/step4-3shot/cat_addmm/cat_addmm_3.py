
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x2):
        v2 = torch.cat([x2, x2, x2, x2, x2, x2, x2, x2], dim=1)
        return v2
        
# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
