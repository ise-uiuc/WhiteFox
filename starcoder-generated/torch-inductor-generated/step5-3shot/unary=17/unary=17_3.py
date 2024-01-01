
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = F.conv_transpose2d(x1, weight=torch.randn(8, 3, 1, 1), bias=torch.randn(8), stride=1, padding=1, output_padding=1)
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
