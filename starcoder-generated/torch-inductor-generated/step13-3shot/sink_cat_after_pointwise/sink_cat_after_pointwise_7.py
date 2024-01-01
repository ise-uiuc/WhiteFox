
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=-1)
        x = y.view(-1).tanh() if x.shape[0] == 1 else y.tanh().view(-1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
