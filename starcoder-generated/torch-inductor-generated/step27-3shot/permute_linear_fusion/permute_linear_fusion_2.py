
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v6 = torch.nn.Hardtanh()(v1)
        v2 = torch.nn.functional.linear(v6, self.linear.weight, self.linear.bias)
        v2 = v2 + v1
        v2 = torch.nn.functional.relu(v2)
        v3 = self.sigmoid(v2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
