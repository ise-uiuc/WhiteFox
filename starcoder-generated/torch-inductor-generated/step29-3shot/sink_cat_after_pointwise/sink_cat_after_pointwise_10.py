
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        v1 = y.view(y.shape[0], -1) if torch.numel(v1) == 1 else v1.view(v1.shape[0], -1)
        v2 = v1.tanh()
        x = v2.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
