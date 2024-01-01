
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=6, out_features=6, bias=False)
        self.other = torch.tensor([0.066956, 0.176144, 0.618862, -0.419607, -0.379710, -0.042437], dtype=torch.float32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
