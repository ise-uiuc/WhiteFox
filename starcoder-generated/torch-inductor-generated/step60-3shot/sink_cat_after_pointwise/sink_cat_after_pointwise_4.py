
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x, x], dim=0)
        x1 = x.reshape(1, -1)
        x2 = torch.relu(x1)
        return x2.view(x2.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(1, 2)
