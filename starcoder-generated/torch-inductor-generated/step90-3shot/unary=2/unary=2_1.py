
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose3d(5, 9, kernel_size=(1, 9, 9), stride=(2, 1, 1))
        self.conv_transpose2 = torch.nn.ConvTranspose3d(9, 6, kernel_size=(5, 4, 6), stride=(3, 4, 5))
        self.conv_transpose3 = torch.nn.ConvTranspose3d(6, 7, kernel_size=(1, 5, 1), stride=(4, 5, 2))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose2(v9)
        v11 = v10 * 1.4534120063248525
        v12 = v11 * 0.2
        v13 = v11 + v12
        v14 = v13 * 0.2928932188145398
        v15 = torch.ceil(v14)
        v16 = torch.max(v15)
        v17 = v16 + 1
        v18 = v17 * 45
        v19 = v13 * v18
        v20 = self.conv_transpose3(v19)
        v21 = v20 + 2.80259692223761e-06
        v22 = v21 * 342.0
        return v22
# Inputs to the model
x1 = torch.randn(9, 5, 10, 5, 4)
