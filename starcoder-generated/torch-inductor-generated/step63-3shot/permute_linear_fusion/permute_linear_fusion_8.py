
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 2)
        self.linear_2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        x2 = self.linear_1(x1)
        x3 = torch.nn.functional.linear(v1, self.linear_2.weight, self.linear_2.bias)
        return {"x2": x2, "x3": x3}
# Inputs to the model
x1 = torch.randn(1, 2, 2)
