
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.flatten = torch.nn.Flatten(0, 1)
        self.softmax = torch.nn.Softmax(dim=0)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v4 = v3.unsqueeze(1)
        v5 = self.conv(v4)
        v6 = v5.squeeze(1)
        v7 = self.flatten(v6)
        v8 = self.softmax(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 2)
