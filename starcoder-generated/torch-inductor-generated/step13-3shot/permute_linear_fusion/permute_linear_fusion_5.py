
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.linear = torch.nn.Linear(2, 2)
        self.leaky_relu = torch.nn.LeakyReLU()
    def forward(self, x1):
        v1 = self.relu6(x1)
        v2 = self.linear(v1)
        v3 = self.leaky_relu(v2)
        return x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
