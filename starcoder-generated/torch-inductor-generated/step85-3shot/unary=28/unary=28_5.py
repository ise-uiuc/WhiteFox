
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=6.0):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x):
        return torch.clamp(
            torch.clamp(
                self.linear(x),
                min=self.min_value),
            max=self.max_value)

# Initializing the model
m = Model(0.0, 6.0)

# Inputs to the model
x = torch.randn(1, 8)
