 output
class Model(torch.nn.Module):
    def __init__(self, min_value: float, max_value: float):
        super().__init__()
        self.linear = torch.nn.Linear(10, 12)
        self.max_output = max_value
        self.min_output = min_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, self.min_output)
        v3 = torch.clamp(v2, max=self.max_output)
        return v3

# Initializing the model
m = Model(0.18707, 0.80180)

# Inputs to the model
x1 = torch.randn(64, 10)
