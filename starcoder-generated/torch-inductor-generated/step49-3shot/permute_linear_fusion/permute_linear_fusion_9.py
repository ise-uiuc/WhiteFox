
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2800, 800)
        self.conv1 = torch.nn.Conv1d(in_channels=10, out_channels=10, kernel_size=(1,))
        self.conv2 = torch.nn.Conv1d(in_channels=10, out_channels=10, kernel_size=(1,))
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(1)
        v4 = self.conv2(v3)
        v5 = v4.squeeze(1)
        v6 = self.conv1(v1)
        x2 = v5 + v6
        return x2
# Inputs to the model
x1 = torch.randn(1, 2800, 10)
