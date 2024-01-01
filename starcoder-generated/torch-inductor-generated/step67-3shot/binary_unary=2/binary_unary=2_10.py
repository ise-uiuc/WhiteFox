
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 128, 2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - torch.Tensor([[[[ 0.8361],
                                   [ 0.2916],
                                   [-1.1684],
                                   [-0.0681]]]])
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 10, 10)
