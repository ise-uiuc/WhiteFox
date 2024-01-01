
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=0)
        if y.shape[1] == 6:
            y = y.mean(dim=-1)
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
