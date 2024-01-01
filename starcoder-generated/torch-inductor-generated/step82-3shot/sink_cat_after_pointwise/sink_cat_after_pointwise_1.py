
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(2, 1, 3, -1)
        return torch.relu(x)
# Inputs to the model
x = torch.randn(2, 3, 4)
