
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 	self.clamp_min = torch.nn.modules.activation.ReLU(inplace=True)
        self.clamp_max = torch.jit.script(torch.clamp)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.clamp_min(v1, min_value=-1000.0)
        v3 = self.clamp_max(v2, max_value=-194.3)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
