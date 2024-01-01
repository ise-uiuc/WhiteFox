
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = torch.unsqueeze(x1, 1)
        x3 = torch.transpose(x2, 1, -1)
        x4 = torch.squeeze(x1, -2)
        x5 = torch.matmul(x3, x4)
        x6 = (x3) * (x4)
        x7 = (x3) + (x4)
        v1 = torch.nn.functional.linear(x5, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 2)
