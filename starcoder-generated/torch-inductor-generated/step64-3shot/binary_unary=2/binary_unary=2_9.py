
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = v1 - 2
        v3 = F.relu(v2)
        return F.relu(v3)
# Inputs to the model
x1 = torch.randn(1, 16, 4, 4)
