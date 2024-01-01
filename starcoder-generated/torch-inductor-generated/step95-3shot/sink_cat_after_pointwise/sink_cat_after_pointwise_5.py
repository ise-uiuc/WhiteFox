
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod1 = torch.nn.ReLU()
        self.mod2 = torch.nn.Tanh()
    def forward(self, x):
        y = torch.cat((x.tanh(), x.tanh(), x.tanh()), dim=1)
        return y.view(y.shape[0], -1).relu() if y.shape!= (1, 3) else y.view(y.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
