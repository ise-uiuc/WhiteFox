
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
    def forward(self, x0):
        v1 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v2 = torch.sigmoid(v1)
        v3 = v2.squeeze(1)
        return v3.permute(0, 2, 1)
# Inputs to the model
x0 = torch.randn(1, 2, 2)
