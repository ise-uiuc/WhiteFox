
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.randn(128, 512), torch.randn(512))
        v2 = torch.clamp(v1 + 3, 0, 6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 512)
