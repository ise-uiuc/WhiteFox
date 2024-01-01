
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(3, 2, 0)
        v3 = torch.transpose(v2, 0, 1)
        v4 = torch.mul(v3.permute(1, 0, 2), (1, v3, v2))
        return v2
# Inputs to the model
x1 = torch.randn(3, 2, 2, 3, 4, device=torch.device("cuda"))
