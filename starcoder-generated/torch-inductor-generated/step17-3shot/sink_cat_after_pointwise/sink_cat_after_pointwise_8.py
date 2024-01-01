
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = torch.relu(x)
        return torch.cat((x, z), dim=1)
# Inputs to the model
x = torch.randn(4)
