
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), dim=1)
        v2 = torch.cat((x1, x2), dim=1)
        v8 = torch.cat((v1, v2), dim=1)
        y = torch.relu(v8)
        v3 = torch.cat((v1, y), dim=1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 3)
