
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.7976931348623157e+308, max_value=1.7976931348623157e+308):
        super().__init__()
        self.linears = torch.nn.Linear(3, 8, bias=True)
        self.clamps = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Identity()
        )
        self.clamps[0].clamp_min = min_value
        self.clamps[1].clamp_min = min_value
 
    def forward(self, x1):
        v1 = self.linears(x1)
        v2 = self.clamps(v1)
        v3 = self.clamps(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
