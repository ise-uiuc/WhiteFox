
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # PyTorch's quantized conv2d does not support groups, so we add the
        # groups attribute during model construction and remove it before
        # exporting a model.
        self.conv2d = nn.Conv2d(1, 20, 5, 1, groups=1)
        for p in self.conv2d.parameters():
            # weight
            p.data.copy_(-0.1 + torch.rand(p.numel()))
            # bias
            p.data.copy_(torch.rand(p.numel()))

        self.bn2d = nn.BatchNorm2d(20)
    def forward(self, x):
        x1 = self.conv2d(x)
        x2 = self.bn2d(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
