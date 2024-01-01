
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
 
        torch.nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=0.00381269)
        v3 = torch.clamp_max(v2, max_value=3.0708757648026783e-06)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
