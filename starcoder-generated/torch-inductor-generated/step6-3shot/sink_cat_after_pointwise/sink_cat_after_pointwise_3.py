
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = y.relu().tanh()
        y = y.view(-1, 1, y.shape[1]).permute(1, 0, 2).reshape(2, -1)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
