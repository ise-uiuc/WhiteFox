
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.sin(x)
        y = y.clone().detach()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
