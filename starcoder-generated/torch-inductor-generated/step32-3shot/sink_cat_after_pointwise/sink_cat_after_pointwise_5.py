
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        x = y.view(x.shape[0], -1).tanh() if y.shape!= torch.Size([64, 24]) else y.view(x.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4, 3, 5, 3)
