
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(4, 4))
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(4, 4))
        self.flatten = Flatten()
        self.linear = torch.nn.Linear(16, 2)

    def forward(self, x1):
        v1 = torch.nn.functional.relu(self.conv1(x1))
        v2 = torch.nn.functional.relu(self.conv2(v1))
        v3 = torch.max(v2, dim=-1)[0]
        v4 = self.flatten(v3)
        v5 = torch.nn.functional.relu(self.linear(v4))
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
