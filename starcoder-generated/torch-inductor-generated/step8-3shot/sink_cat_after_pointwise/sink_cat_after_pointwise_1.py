
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 6)
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        z = self.weight.view(-1, 6)
        x = torch.cat((y, z), dim=1).view(y.shape[0], -1).tanh()
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
