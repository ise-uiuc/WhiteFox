
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=36, out_features=10, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 5
        v3 = v2.clamp(0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 36)
