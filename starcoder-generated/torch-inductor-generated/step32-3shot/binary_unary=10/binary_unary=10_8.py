
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=10, out_features=10)
        self.linear2 = torch.nn.Linear(in_features=10, out_features=1)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        other = torch.rand(1, 10)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
