
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=0)
        x = torch.tanh(y)
        x = x.view(2 * x.shape[0], -1)
        x = x.view(-1, 3 * x.shape[1])
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3)
