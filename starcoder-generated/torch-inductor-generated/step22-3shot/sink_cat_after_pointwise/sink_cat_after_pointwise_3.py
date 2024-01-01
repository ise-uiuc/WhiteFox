
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.abs(x)
        x2 = torch.cat((x1, x1), dim=1)
        return x2.reshape(x.shape[0], -1)
# Inputs to the model
x = torch.randn(1, 2, 2)
