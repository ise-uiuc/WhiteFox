
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.clamp_min(x.clone().fill_(1), min_value=x.clone().fill_(0), out=None)
        v3 = v1.clamp_max(x.clone().fill_(0), max_value=x.clone().fill_(2), out=None)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
