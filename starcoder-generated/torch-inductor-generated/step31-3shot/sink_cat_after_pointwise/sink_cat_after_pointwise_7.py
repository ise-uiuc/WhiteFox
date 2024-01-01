
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y, z = x.view(x.shape[0], -1), x.view(x.shape[0], -1)
        w = torch.relu(z)
        return torch.cat((x,y), dim=1).tanh()
# Inputs to the model
x = torch.randn(2, 32, 32, 32)
