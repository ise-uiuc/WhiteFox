
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x.clone(), x.clone()), dim=1)
        y = x.permute(1, 0, 2)
        return y.tanh()
# Inputs to the model
x = torch.randn(3, 2, 4)
