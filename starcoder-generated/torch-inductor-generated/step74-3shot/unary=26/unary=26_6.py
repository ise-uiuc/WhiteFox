
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(138, 57, (7, 12), stride=(3, 4), bias=False)
    def forward(self, x65):
        x1 = self.conv_t(x65)
        x2 = x1 > 0
        x3 = x1 * 8.1
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.relu(x4)
# Inputs to the model
x65 = torch.randn(19, 138, 29, 30)
