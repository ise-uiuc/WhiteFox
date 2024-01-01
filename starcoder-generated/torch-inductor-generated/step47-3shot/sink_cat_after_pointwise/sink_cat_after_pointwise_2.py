
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(1, 2, 1, 16) if x.size(-1) == 16 else x.view(1, 2, 2, 8)
        return x.view(x.size(0), -1)
# Inputs to the model
x = torch.randn(1, 2, 2, 16)
