
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=20, out_features=50)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 - 3.5
        x4 = torch.relu(x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 20)
