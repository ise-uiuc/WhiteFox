
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        t1 = x1.permute(-1, 0, 2, 3)
        t2 = self.linear(t1)
        t3 = t2.transpose(-2, -1)
        t4 = torch.matmul(t1, t3)
        return torch.nn.functional.relu(t4)
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
