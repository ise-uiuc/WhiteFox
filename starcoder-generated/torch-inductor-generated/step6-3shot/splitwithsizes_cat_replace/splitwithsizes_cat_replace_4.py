
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x2):
        v9, v5, v8 = torch.split(x2, [9, 5, 8], dim=1)
        v0 = torch.cat([v9, v5, v8], dim=1)
        return v0
        
# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 14, 5, 5)
