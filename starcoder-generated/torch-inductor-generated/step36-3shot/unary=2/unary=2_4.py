
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, (1, 1))
        self.conv_transpose11 = torch.nn.ConvTranspose2d(3, 1, (1, 1))
        self.conv_transpose22 = torch.nn.ConvTranspose2d(3, 1, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=3, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.25
        v3 = v1 * v1
        v4 = v3 * 0.1
        v5 = v1 + v4
        v6 = v5 * 0.3
        v7 = torch.relu(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose11(v1)
        v11 = v10 * 0.25
        v12 = v10 * v10
        v13 = v12 * 0.1
        v14 = v10 + v13
        v15 = v14 * 0.3
        v16 = torch.relu(v15)
        v17 = v16 + 1
        v18 = v11 * v17
        v19 = self.conv_transpose22(v10)
        return v19
# Input to the model
x1 = torch.randn(1, 3, 4, 6) 
