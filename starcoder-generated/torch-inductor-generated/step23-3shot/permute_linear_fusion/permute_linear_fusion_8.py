
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        x4 = torch.nn.functional.relu(v4)
        v5 = x4.detach()
        v6 = torch.argmax(v5, dim=-1)
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 4, 5, 5)
