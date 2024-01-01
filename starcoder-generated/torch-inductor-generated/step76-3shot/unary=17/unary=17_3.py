
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_transpose_1 = torch.nn.ConvTranspose1d(256, 2000, 1)
        self.pointwise_transpose_2 = torch.nn.ConvTranspose1d(2000, 600, 10)
        self.pointwise_transpose_3 = torch.nn.ConvTranspose1d(600, 240, 10)
        self.pointwise_transpose_4 = torch.nn.ConvTranspose1d(240, 48, 10)
    def forward(self, x1):
        v1 = self.pointwise_transpose_1(x1)
        v2 = torch.relu(v1)
        v3 = self.pointwise_transpose_2(v2)
        v4 = torch.relu(v3)
        v5 = self.pointwise_transpose_3(v4)
        v6 = torch.relu(v5)
        v7 = self.pointwise_transpose_4(v6)
        v8 = torch.relu(v7)
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 256, 375)
