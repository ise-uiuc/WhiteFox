
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(-1)
        x = torch.cat([x, x], dim=0)
        x = torch.relu(x)
        x = x.view(-1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
