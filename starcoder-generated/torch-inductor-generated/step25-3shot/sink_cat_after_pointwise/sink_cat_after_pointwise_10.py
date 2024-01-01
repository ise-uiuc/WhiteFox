
class Model(torch.nn.Mo):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x, x), dim=1)
        y = y.view(y.shape[0], -1)
        z1 = torch.tanh(y)
        z2 = torch.relu(y)
        y = z1
        y = y.view_as(z2)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
