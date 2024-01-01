
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 16)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(32, 64)
        self.conv = torch.nn.Conv2d(3, 6, kernel_size=(1, 1), stride=(1, 1))
        self.max_pooling2d = torch.nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.permute(0, 2, 1)
        v3 = self.flatten(v1)
        v4 = v3.reshape(v1.shape[0], 32, 1, 1)
        v5 = self.linear1(v3)
        v6 = v3.reshape(v5.shape[0], 6, 2, 2)
        v7 = self.conv(v6)
        v8 = self.relu(v3)
        v9 = self.max_pooling2d(v8)
        return v1.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
