
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose= torch.nn.ConvTranspose3d(3, 2, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), output_padding=(1, 1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.relu(v2)
        return torch.sigmoid(v2)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
