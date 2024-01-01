
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=128, out_features=32, bias=True)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        # Please edit the values of the keyword arguments (min_value and max_value).
        v2 = torch.clamp_min(v1, min_value=0.0)
        v3 = torch.clamp_max(v2, max_value=10.0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
