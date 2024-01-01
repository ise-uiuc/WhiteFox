
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x, x), dim)
        y = torch.relu(y)
        y = torch.reshape(y, (-1, 8))
        y = y.mean(dim=0)
        return y
# Inputs to the model
x = torch.randn(1, 3, 4)
