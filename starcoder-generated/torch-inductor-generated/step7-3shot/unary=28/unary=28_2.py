
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(13, 11)
        self.min = 2.1
        self.max = 5.5
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=self.min)
        v3 = torch.clamp_max(v2, max_value=self.max)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 13)
