
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 6)
        self.linear2 = torch.nn.Linear(6, 1)
 
    def forward(self, x1, min_value, max_value):
        v1 = self.linear1(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(6, 4)
min_value = 0.0000001
max_value = 0.9999999
