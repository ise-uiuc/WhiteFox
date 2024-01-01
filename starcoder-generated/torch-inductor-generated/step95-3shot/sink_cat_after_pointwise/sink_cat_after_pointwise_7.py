
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x.tanh(), x, x.tanh()), dim=-1) # y is non-overlapping
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
