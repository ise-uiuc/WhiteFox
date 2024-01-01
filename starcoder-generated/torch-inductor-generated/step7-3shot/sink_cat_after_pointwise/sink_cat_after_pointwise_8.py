
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.add(x, x)
        v2 = torch.add(x, x)
        y = torch.cat((v1, v2), dim=0)
        y = torch.nn.functional.relu(y)
        return y
# Inputs to the model
x = torch.randn(2, 3)
