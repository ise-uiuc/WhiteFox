
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(**kwargs)
        self.min = -kwargs['out_features']
        self.max = kwargs['out_features']

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max, out=v2)
        return v3

# Initializing the model
m = Model(in_features=30, out_features=5)

# Inputs to the model
x1 = torch.randn(1, 30)
