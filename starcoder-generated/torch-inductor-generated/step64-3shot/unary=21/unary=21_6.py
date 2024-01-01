
class ModelTanh(torch.nn.Module):
    def __init__(self):
        self.block1 = block1 = models.conv1x1(1, 1, stride=1)
        self.block2 = block2 = models.conv1x1(256, 256, stride=1)
        self.block3 = block3 = models.conv1x1(1, 1, stride=1)
    def forward(self, x):
        x = x.float()
        x = self.block1(x)
        x = F.gelu(x)
        x = self.block2(x)
        x = F.gelu(x)
        x = self.block3(x)
        return x
