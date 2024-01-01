
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v10 = torch.nn.ReLU()(v6)
        v9 = v2 * v10
        v12 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)(v9)
        v11 = self.conv_transpose2(v12)
        v13 = v11 * 0.5
        v14 = v11 * v11 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v19 = torch.nn.Tanh()(v17)
        v20 = v19 + 1.0
        v18 = v13 * v20
        v21 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=1)(v18)
        v22 = self.conv_transpose3(v21)
        return v22
# Inputs to the model
x1 = torch.randn(1, 4, 12, 14)
