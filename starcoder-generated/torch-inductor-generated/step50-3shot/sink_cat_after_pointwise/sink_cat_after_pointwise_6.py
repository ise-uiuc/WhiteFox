
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([torch.relu(x)], dim=1).view(-1)
# Inputs to the model
x = torch.randn(2, 4, 4)
