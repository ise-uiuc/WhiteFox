
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
 
    def forward(self, input_tensor):
        v1 = self.linear(input_tensor)
        v2 = v1 * torch.clamp(min=0.0, max=6.0, v1 + 3.0)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(6)
