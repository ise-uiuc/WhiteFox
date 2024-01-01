
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = torch.relu(x)
        return x.view(-1)
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
