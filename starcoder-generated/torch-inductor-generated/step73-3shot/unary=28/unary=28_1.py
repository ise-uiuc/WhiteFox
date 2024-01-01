
class Model(torch.nn.Module):
    def __init__(self, min_value=-1, max_value=1):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, x2)
        v2 = torch.clamp_min(v1, self.min_value)
        return torch.clamp_max(v2, self.max_value)
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
x2 = torch.randn(10, 512)
