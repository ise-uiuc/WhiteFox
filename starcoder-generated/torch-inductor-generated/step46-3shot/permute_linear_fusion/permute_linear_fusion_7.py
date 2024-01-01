
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Identity() # Identity is a special case of sigmoid function
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.sigmoid(v2)
        v4 = v3.squeeze(-1)
        v5 = v4.transpose(1, 2)
        v6 = torch.sum(v5, dim=1, keepdim=True)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
