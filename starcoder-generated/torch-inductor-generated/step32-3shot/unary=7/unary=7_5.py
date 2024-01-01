
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torchvision.ops.misc.clamp_by_value(v1 + 3, 0.0, 6.0)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 32, 3, 3)
