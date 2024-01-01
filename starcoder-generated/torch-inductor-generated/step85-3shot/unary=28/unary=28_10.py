
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.fc = torch.nn.Linear(768, 768)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.clamp_min(v1, min)
        v3 = torch.clamp_max(v2, max)
        return v3

# Initializing the model

# Inputs to the model
__minimum_value__ = -1.0
__maximum_value__ = 1.0
x = torch.rand(768)
m = Model(-1.0, 1.0)
