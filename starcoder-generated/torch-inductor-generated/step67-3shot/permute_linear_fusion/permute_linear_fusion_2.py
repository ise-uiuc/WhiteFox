
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.relu(v2)
        v4 = (v3 + v1).detach()
        v5 = torch.sum(v4, dim=(1,))
        return v2 + v5.unsqueeze(0).to(v2.dtype)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
