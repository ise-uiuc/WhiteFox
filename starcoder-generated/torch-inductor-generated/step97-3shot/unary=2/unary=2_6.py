
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 10, 1, stride=1, padding = 0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 12, 2, stride=(3, 2), dilation=2, padding=2)
        self.max_pool = torch.nn.MaxPool2d(3, stride=1, padding=0)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(12, 14, 4, stride=(2, 3), dilation=2, padding=1)
        self.max_pool2 = torch.nn.MaxPool2d(3, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
        self.conv_transpose4 = torch.nn.ConvTranspose2d(14, 16, 8, stride=(2, 3), dilation=2, padding=4)
    def forward(self, x1):
      v1 = self.conv_transpose(x1)
      v2 = v1 * 0.5
      v3 = v1 * v1 * v1
      v4 = v3 * 0.044715
      v5 = v1 + v4
      v6 = v5 * 0.7978845608028654
      v7 = torch.tanh(v6)
      v8 = v7 + 1
      v9 = v2 * v8
      v10 = self.conv_transpose2(v9)
      v11 = v10 * 1469.3408203125
      v12 = self.max_pool(v11)
      v13 = self.conv_transpose3(v12)
      v14 = v13 * 0.601315
      v15 = v14 + torch.max(v13, (2, 3), True)[0]
      v16 = self.max_pool2(v15)
      v17 = self.relu(v16)
      v18 = self.conv_transpose4(v17)
      return v18
# Inputs to the model
x1 = torch.randn(2, 1, 180, 180)
