
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat((x1, x1), dim=0)
        v2 = torch.relu(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3)
