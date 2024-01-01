
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 32, in_features=128, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=0.5)
        v3 = torch.clamp_max(v2, max_value=0.707)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 128)
