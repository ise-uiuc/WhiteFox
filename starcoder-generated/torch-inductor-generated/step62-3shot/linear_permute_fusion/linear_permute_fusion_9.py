
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = torch.nn.Linear(3, 3).to(torch.float16)
        self.conv2d = torch.nn.Conv2d(3, 3, (1, 3)).to(torch.float16)
        self.pooling2d = torch.nn.MaxPool2d(4, 4)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear2.weight, self.linear2.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = self.pooling2d(v2)
        v4 = self.conv2d(v3)
        v5 = torch.nn.functional.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
