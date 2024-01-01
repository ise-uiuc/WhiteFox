
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        x1 = torch.cat([x, x], dim=1).view(x.shape[0], -1)
        x2 = torch.cat([x1, x1], dim=1).view(x.shape[0], -1)
        x = torch.relu(x2)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
