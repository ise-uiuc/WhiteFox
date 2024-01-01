
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.transpose(x1, 0, 1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = torch.squeeze(v3, 0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 15, 120, 120)
