
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        x = torch.cat((x, x), dim=1)
        x = x.view(x.shape[0], -1).tanh()
        x.relu()
        return x
# Inputs to the model
x = torch.randn(1, 2, 1)
# 