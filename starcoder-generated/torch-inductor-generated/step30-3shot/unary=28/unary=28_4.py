
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x1, **kwargs):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, max(kwargs['min_value'], v2.min()))
        v3 = torch.clamp_max(v2, kwargs['max_value'])
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64)
