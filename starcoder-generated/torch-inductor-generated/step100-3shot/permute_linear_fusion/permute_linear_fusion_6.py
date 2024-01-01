
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.relu6.training = False
        self.linear = torch.nn.Linear(4, 16)
        self.transpose = torch.transpose
        self.matmul = torch.matmul
    def forward(self, x1):
        v1 = self.relu6(x1)
        v1 = v1.to(self.linear.weight.dtype)
        v1 = self.transpose(v1, -2, -1)
        v1 = self.matmul(v1, self.linear.weight)
        return v1
# Inputs to the model
x1 = torch.randn(1, 4)
