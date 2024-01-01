
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.transpose = torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=2, output_padding=3)
        self.max_pool = torch.nn.AvgPool2d(2, stride=2)
    def forward(self, x1):
        v1 = self.sigmoid(x1)
        v2 = self.transpose(v1)
        v3 = self.max_pool(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 142, 153)
