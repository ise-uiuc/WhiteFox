
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear((1, 20), (10,))
        self.linear_1 = torch.nn.Linear((10,), 1)
        self.conv = torch.nn.Conv2d(in_channels=(1, 20), out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(1)
        v4 = self.conv(v3)
        v4 = v4.squeeze(1).permute(0, 2, 1)
        v5 = torch.nn.functional.linear(v4, self.linear_1.weight, self.linear_1.bias)
        v5 = v5.unsqueeze(1)
        v6 = x1.unsqueeze(2)
        v7 = torch.sub(v5, v1)
        v8 = torch.nn.functional.linear(v7, self.linear.weight, self.linear.bias)
        v9 = torch.nn.functional.linear(v8, self.linear.weight, self.linear.bias)
        v10 = torch.mul(v9, v10, v10, v10)
        v10 = torch.reshape(v10, (-1, v10.size()[-2] * v10.size()[-1]))
        v11 = torch.norm(v10, p=2, dim=1)
        return v11
# Inputs to the model
x1 = torch.randn(1, 20, 1)
