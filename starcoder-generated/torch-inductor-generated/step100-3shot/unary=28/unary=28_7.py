
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128,256)
 
    def forward(self, x2, min_value=0, max_value=13):
        v1 = self.linear(x2)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initialize the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 128)
