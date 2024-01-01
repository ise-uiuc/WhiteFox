
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 224)
 
    def forward(self, x1, kwargs):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=kwargs["min"])
        v3 = torch.clamp_max(v2, max_value=kwargs["max"])
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64 * 64, 224)
kwargs = {
    "min": -1.0,
    "max": 1.0
}
