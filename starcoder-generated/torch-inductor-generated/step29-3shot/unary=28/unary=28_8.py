
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, input, min_value, max_value):
        v1 = self.linear(input)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3
 
# Initializing the model with arguments
m = Model()

# Inputs to the model
x1 = torch.randn(1000, 16)
