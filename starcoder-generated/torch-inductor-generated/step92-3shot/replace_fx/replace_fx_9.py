
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if torch.rand(1) > 0.5:
            x = torch.clamp(x, min=0)
        else:
            x = x - 0.5
        x = torch.clamp(x, max=1)
        x = x + 0.5
        return x
# Inputs to the model
x = torch.randn(2, 2)
