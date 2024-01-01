
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=100, out_features=64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
