
class Model(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, input, **kwargs):
        v1 = self.t1(input)
        v2 = torch.clamp_min(v1, kwargs["min_value"])
        v3 = torch.clamp_max(v2, kwargs["max_value"])
        return v3

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 3, 64, 64)
