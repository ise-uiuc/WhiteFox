
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(472, 477)
        self.min_value = torch.tensor(-0.20202)
        self.max_value = torch.tensor(0.002711)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp(v1, self.min_value, self.max_value)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(100, 472)
