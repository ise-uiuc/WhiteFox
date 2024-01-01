
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = F.conv2d
    def forward(self, x1):
        v1 = self.conv(input=x1, weight=torch.Tensor())
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
