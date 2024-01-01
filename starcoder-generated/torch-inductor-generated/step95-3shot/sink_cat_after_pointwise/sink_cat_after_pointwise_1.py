
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x.tanh(), x.tanh(), x.tanh()), dim=1)
        return y.view(-1) if y.shape!= (1, 3) else y.view(1, 1)
# Inputs to the model
x = torch.randn(2, 4)
