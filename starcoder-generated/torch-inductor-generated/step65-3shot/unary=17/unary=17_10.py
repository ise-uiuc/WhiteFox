
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), padding=(1, 1))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), padding=(1, 0))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v1 = torch.relu(v1)
        v2 = self.conv_transpose2(x1)
        v2 = torch.sigmoid(v2)
        v3 = self.conv_transpose3(x1)
        v3 = torch.relu(v3)
        v4 = torch.relu(v1)
        v5 = torch.sigmoid(v4)
        return torch.sigmoid(v5)
# Inputs to the model
x1 = torch.randn(1, 1, 2, 3)
