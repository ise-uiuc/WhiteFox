
class model_relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=5, stride=3, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = F.relu(v1)
    return v2
# Inputs to the model
x = torch.randn(2, 1, 288, 288)
