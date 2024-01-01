
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, kernel_size=(2, 3), stride=(2, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return torch.squeeze(v3, dim=0)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 25)
