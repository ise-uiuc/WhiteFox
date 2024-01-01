
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 4096, bias=True)
 
    def forward(self, input_tensor, min_value, max_value):
        v1 = self.linear(input_tensor)
        return torch.clamp_min(torch.clamp_max(v1, max_value), min_value)

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 1024)
