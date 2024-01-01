
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.flatten = torch.nn.Flatten(0, 1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(1)
        v4 = self.conv(v3)
        v5 = v4.squeeze(1)
        v6 = self.flatten(v5)
        v7 = self.softmax(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
