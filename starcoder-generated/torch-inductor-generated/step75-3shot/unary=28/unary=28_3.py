
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(128, 16)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=min_value)
        v3 = torch.clamp_max(v2, max_value=max_value)
        return v3

# Initializing the model
__min_value__ = __random__.random() * 10
__max_value__ = __random__.random() * 10
m = Model(__min_value__, __max_value__)

# Inputs to the model
x1 = torch.randn(128)
