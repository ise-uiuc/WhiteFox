
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.reshape(x.shape[0], -1).tanh()
        x = y[:, 0].view(y.shape[0], -1).tanh()

        y = torch.randn(y.size())
        x = torch.cat((x, x), dim=1).tanh()
        return torch.relu(x)
# Inputs to the model
x = torch.randn(2, 2, 2)
