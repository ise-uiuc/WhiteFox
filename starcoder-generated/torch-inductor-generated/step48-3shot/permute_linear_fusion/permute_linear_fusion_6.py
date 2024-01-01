
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Softmax(dim=0)
        self.flatten = torch.nn.Flatten(0, 1)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.softmax(v1)
        v3 = self.flatten(v2)
        v4 = self.relu6(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
