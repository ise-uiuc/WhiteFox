
class Model(torch.nn.Module):
    def __init__(self, min_value=0., max_value=1.):
        super().__init__()
        self.fc = torch.nn.Linear(9, 3)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, min=self.min_value)
        v3 = torch.clamp_max(v2, max=self.max_value)
        return v3

# Initializing the model with the specified arguments
m = Model(min_value=0., max_value=1.)

