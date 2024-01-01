
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.sigmoid(torch.nn.functional.linear(v1, self.linear.weight))
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
