
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 64)
 
    def forward(self, x1, _max_value=-1, _min_value=-1):
        max_value = _max_value
        min_value = _min_value
        v0 = x1.flatten(1)
        v1 = self.fc(v0)
        v2 = v1.clamp_min(min_value)
        v3 = v2.clamp_max(max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
