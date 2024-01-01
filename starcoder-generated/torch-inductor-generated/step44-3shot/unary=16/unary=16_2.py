
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x

# Initializing the model
m = Model() 

# Inputs to the model
x = torch.randn(1, 3)
