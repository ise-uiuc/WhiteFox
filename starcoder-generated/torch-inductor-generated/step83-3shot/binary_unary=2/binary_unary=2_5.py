
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = F.conv1d(x1, torch.randn(3,3,20), stride=1, padding=0)
        v2 = v1 - 1
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 200)
