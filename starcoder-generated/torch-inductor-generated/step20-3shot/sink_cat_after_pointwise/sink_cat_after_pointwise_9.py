
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        x1 = torch.cat((y, y), dim=1)
        x2 = x1.tanh() if y.shape[0] == 1 else x1.tanh()
        x2 = x.reshape(-1).tanh()
        x = x1.sub(x2)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
