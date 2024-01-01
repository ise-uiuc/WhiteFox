
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_17 = torch.nn.ConvTranspose2d(87, 20, kernel_size=1, stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_transpose_18 = torch.nn.ConvTranspose2d(22, 5, kernel_size=1, stride=1, padding=0, bias=False)
        self.concat_7 = torch.nn.ConcatTable()
        self.conv_transpose_19 = torch.nn.ConvTranspose2d(22, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_transpose_20 = torch.nn.ConvTranspose2d(2, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.concat_8 = torch.nn.ConcatTable()
        self.linear_13 = torch.nn.Linear(4, 81)
        self.linear_14 = torch.nn.Linear(81, 3)
    def forward(self, x1, x2):
        v1 = self.conv_transpose_17(x1)
        v2 = self.conv_transpose_18(v1)
        v3 = self.concat_7([x2, v2])
        v4 = torch.sigmoid(v3)
        v5 = self.conv_transpose_19(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv_transpose_20(v6)
        v8 = self.concat_8([v3, v7])
        v9 = torch.sigmoid(v8)
        v10 = self.linear_13(v9)
        v11 = self.linear_14(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 87, 7, 7)
x2 = torch.randn(1, 22, 1, 1)
