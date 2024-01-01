
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_0 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.batch_norm_0 = torch.nn.BatchNorm2d(64, eps=9.999999747378752e-06, momentum=0.8)
    def forward(self, x):
        x = self.conv2d_0(x)
        x = self.batch_norm_0(x)
        x = F.relu(x)
        x = F.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 16, 16)
