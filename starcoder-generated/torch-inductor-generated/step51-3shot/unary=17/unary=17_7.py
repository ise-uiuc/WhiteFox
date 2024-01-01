
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 3, (3, 3), stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 3, (3, 3), stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 1, 50, 50)
