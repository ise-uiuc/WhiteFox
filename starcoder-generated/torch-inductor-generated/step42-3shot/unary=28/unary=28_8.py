
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x):
        v = self.linear(x)
        v = torch.clamp_min(v, -0.8_f32)
        v = torch.clamp_max(v, 0.8_f32)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
