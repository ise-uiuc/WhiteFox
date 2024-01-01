
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v0 = torch.cat([x1, x1], dim=0)
        v1 = v0.permmute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        x3 = x2[-1::-1]
        x4 = torch.nn.functional.max_pool1d(x3, x3.shape[2])
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
