
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.relu().tanh().view(-1).view(1, -1)
        x = y if y.shape == x.shape else torch.cat((y, y), dim=1)
        x = x[..., -1]
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
