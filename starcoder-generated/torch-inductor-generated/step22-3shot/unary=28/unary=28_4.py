
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = F.relu6(v1)
        v3 = F.hardtanh(v1)
        v4 = nn.Hardsigmoid()(v1)
        v5 = torch.round(v1)
        v6 = F.hardswish(v1)
        return v6

# Initializing the model
m = Model(min_value=0, max_value=1)

# Inputs to the model
x1 = torch.randn(1, 3)
