
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v7 = self.linear.weight * 0
        v2 = torch.nn.functional.linear(v1, self.linear.weight + 1, self.linear.bias, self.linear.bias)
        v7 = v2 - v7
        x2 = torch.nn.functional.relu(v2)
        v5 = torch.cat((1*v7, 1*torch.ones_like(x1)), dim=1)
        v3 = torch.nn.functional.softmax(v5, dim=-1) * (1-torch.nn.functional.relu(1-x2))
        v6 = torch.cat((torch.cat((1*x1, 1*v3), dim=1), 1*torch.ones_like(v3)), dim=1)
        v4 = torch.einsum("abi, bci->bc", v6.permute((0, 2, 1)), v6)
        v4 = v4 - v6
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
