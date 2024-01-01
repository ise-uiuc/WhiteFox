
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(3, 16, kernel_size=(1, 4, 4), padding=2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(16, 8, kernel_size=(1, 7), padding=3, stride=1)
        self.relu = torch.nn.ReLU()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 3, kernel_size=(1, 4), padding=1, stride=1)
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v2 = self.conv_transpose(v1)
        v3 = self.conv_transpose1(v2)
        v4 = self.relu(v3)
        v5 = self.conv_transpose2(v4)
        return torch.sigmoid(v5)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 64, 64)
