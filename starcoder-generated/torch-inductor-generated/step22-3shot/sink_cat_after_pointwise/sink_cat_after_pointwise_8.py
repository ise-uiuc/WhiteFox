
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=-1)
        y = y.view(y.shape[0], -1)
        y.tanh()
        x = y.view(y.shape[0], -1).tanh() if torch.numel(y) == 1 else y.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
