
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        s = x.shape
        w = x.view(1, -1)
        y = torch.cat((w, w), dim=1)
        return x
# Inputs to the model
x = torch.rand((2, 3, 4))
