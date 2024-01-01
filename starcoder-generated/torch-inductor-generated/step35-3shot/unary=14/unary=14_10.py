
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv2d_1(x1)
        v1 = v1.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True) # Global average pooling layer
        v1 = torch.sigmoid(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
