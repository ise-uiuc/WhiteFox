
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.embedding = torch.nn.Embedding(16, 32)
    def forward(self, x1):
        v1 = self.embedding(x1)
        v2 = v1 * v1
        v3 = self.embedding(x1)
        v3 = v3 + v2
        v4 = v1 + v3
        v4 = v4 / 2
        v5 = v3.permute(0, 3, 1, 2)
        v6 = self.conv(v5)
        v7 = self.embedding(x1)
        v7 = v7 * 4
        v8 = torch.nn.functional.linear(v7, self.linear.weight, self.linear.bias)
        v9 = v6 + v8
        v10 = torch.nn.functional.relu(v9)
        return self.linear(v10)
# Inputs to the model
x1 = torch.randint(0, 16, (1,))
