
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = torch.relu(y)
        z = torch.cat((y + 1, y + 1), dim=1)
        y = z.view(z.shape[0], -1).tanh()
        return y
# Inputs to the model
x = torch.randn(2, 2, 2)
