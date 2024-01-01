
class ExampleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.clamp(torch.clamp(x1, min=0.0), max=1.2)
        x3 = torch.clamp(torch.clamp(x1, min=0.0), max=1.2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
