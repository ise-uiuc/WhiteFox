
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1, False)
 
    def forward(self, x1, min_value=1, max_value=2):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        return torch.clamp_max(v2, max_value)
       
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
