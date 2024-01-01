
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.concat((x, x), dim=1)
        b = a.reshape(x.shape[0], -1)
        return torch.relu(b)
# Inputs to the model
x = torch.randn(5, 3, 4)
