
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(64, 64, bias=False)
 
    def forward(self, x1):
        v1 = self.l(x1.view(x1.size(0), -1))
        v2 = v1 * torch.clamp(v1 + 3, min=-3, max=3)
        v3 = v2 / 6
        return v3
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
