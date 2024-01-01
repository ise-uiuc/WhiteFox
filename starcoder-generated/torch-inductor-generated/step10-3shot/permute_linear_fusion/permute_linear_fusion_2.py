
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).set_weight(torch.rand(2, 2, dtype=torch.float32))
        self.conv = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(2, 2), stride=(1, 1), padding=(0,), dilation=(1,))
        self.convbn = torch.nn.BatchNorm2d(num_features=1).set_weight(torch.rand(1, dtype=torch.float32))
    def forward(self, x1):
        v1 = x1.permute((0, 2, 1))
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.linear(v2, self.conv.weight, self.conv.bias)
        v4 = v3.unsqueeze((1))
        v5 = self.convbn(v4)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
