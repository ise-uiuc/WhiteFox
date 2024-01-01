
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.linear = torch.nn.Linear(8, 3)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(2, 2), padding=(1,), dilation=(1,))
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        v0 = x1
        v1 = self.conv1(v0)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v3 = v1 * v3
        v5 = v3.unsqueeze(1)
        v6 = self.conv2(v5)
        v7 = v6.squeeze(1)
        v8 = self.relu(v7)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
