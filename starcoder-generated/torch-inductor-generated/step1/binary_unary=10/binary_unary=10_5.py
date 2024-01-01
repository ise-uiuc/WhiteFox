
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 11)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.ones_like(v1)
        return torch.relu(v1 + 0.5, v2)
    
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
