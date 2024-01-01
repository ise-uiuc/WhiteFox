
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, c=5, h=6):
        y = x.view(-1, 2*c-1, h)
        x = torch.sqrt(y).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
