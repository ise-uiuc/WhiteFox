
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 43, bias=False)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v3 = v2.clamp(0)
        v4 = v3.clamp(min=max(min(v2, 0), 6))
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 224)
