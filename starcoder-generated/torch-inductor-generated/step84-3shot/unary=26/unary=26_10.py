
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(20, 12, 3, stride=3, padding=1)
    def forward(self, x):
        y = self.conv_t(x)
        tensor = y > 0
        tensor1 = y * -0.3
        tensor2 = torch.where(tensor, y, tensor1)
        return tensor2
# Inputs to the model
x = torch.randn(1, 20, 10, 10)
