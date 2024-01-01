
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(20, 40),
            torch.nn.Linear(40, 10)
        )
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        x2 = self.model(x1)
        return x2, x2.clamp(min=self.min_value), x2.clamp(max=self.max_value)

# Initializing the model
m = Model(-20.0, 10)

# Inputs to the model
x1 = torch.randn(1, 20)
