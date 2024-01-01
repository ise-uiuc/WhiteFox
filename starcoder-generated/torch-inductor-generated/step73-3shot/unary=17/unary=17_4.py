
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.max_pool_1 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.conv_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv_3 = torch.nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)
        self.relu_1 = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.relu1(v1)
        v3 = self.max_pool_1(v2)
        v4 = self.conv_2(v3)
        v5 = self.conv_transpose_1(v4)
        v6 = self.conv_transpose_2(v5)
        v7 = self.conv_3(v6)
        v8 = self.relu_1(v7)
        v9 = self.tanh(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
