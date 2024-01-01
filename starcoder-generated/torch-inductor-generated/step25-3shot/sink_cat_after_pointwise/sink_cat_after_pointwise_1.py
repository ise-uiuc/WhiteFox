
class MyModelClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = (x + x).view(-1).tanh()
        return out
# Inputs to the model
x = torch.randn(2, 3, 4)
