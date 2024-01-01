
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=100, out_features=3000, bias=True)
 
    def forward(self, tensor):
        v1 = self.linear(tensor)
        v2 = torch.zeros((3, 50, 100))
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
tensor = torch.randn(1, 100)
