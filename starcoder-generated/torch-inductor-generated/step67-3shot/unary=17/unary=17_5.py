
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose1 = nn.ConvTranspose2d(3, 5, 3, stride=1, output_padding=1)
        self.transpose2 = nn.BatchNorm2d(5, affine=False)
    def forward(self, x1):
        x1 = self.transpose1(x1)
        x1 = F.relu(x1)
        x1 = self.transpose2(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
