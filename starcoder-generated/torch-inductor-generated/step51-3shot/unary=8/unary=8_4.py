
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d = torch.nn.Conv2d(1, 4, kernel_size=5, padding=1)
    def forward(self, x):
        x = F.relu(self.conv2d(x))
        feature = x.detach()
        feature = torch.flatten(feature, 1)
        return feature
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
