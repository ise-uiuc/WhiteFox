
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(50*50, 10)
 
    def forward(self, x):
        v = self.linear(x)
        v1 = v - 0.5
        return torch.relu(v1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 50*50)
