
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 10)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.relu(v1)
        return v2
        
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)

# Output of the model
