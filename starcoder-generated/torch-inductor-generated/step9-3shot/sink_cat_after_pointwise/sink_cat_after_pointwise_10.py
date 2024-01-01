
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = y.tanh()
        y = y.view(x.shape[0], -1)
        y = y.tanh()
        v1 = torch.cat((y, y), dim=1)
        v2 = v1.view(v1.shape[0], -1) if torch.numel(v1) == 1 else v1.view(v1.shape[0], -1)
        x = v2.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
