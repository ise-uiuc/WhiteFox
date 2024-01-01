
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.tanh()
        z = x.cat((y, y), dim=1).permute(1, 2, 0)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
