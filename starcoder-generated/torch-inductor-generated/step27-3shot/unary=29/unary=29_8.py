
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.1, max_value=0.3):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 16, 1, stride=1, padding=1, bias=True)
        self.max_pool2d_1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)
        self.act_1 = torch.nn.ReLU()
        self.conv_transpose2d_1 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0, bias=True)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)
        self.act_2 = torch.nn.ReLU()
        self.conv_transpose2d_2 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = self.max_pool2d_1(v1)
        v3 = self.act_1(v2)
        v4 = v1 + v3
        v5 = self.conv_transpose2d_1(v4)
        v6 = self.max_pool2d(v5)
        v7 = self.act_2(v6)
        v8 = self.conv_transpose2d_2(v7)
        v9 = torch.clamp_min(v8, self.min_value)
        v10 = torch.clamp_max(v9, self.max_value)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 768, 768)
