
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x * 2
        x = x.permute((2, 3, 1, 0))
        x = torch.cat((x, x), dim=4)
        y = x.view(3, 2, 3)
        return torch.relu(y)
# Inputs to the model
x = torch.randn(5, 3, 4)
