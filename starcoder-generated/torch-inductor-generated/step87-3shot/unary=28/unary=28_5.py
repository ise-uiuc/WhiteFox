
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, min_value, max_value):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1,-10)
        v3 = torch.clamp_max(v2,20) 
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 3)
min_value = -10.090442657470703
max_value = 20.080561637878418
