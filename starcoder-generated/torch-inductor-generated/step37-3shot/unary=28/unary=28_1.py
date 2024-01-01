
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, __input__, min_value, max_value):
        v1 = self.linear(__input__)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 128)
min_value = 0.5
max_value = 1

