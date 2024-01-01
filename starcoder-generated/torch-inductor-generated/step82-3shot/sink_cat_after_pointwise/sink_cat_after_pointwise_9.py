
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = 1 + x.view(-1, 2*(1*3)).relu().tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
