
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.softmax(v2, dim=1)[:, 1]
        q = v3.reshape(2, 2).t()
        w = torch.addmm(self.linear.bias, v1, self.linear.weight.t())
        y = torch.nn.functional.sigmoid(w.view(2, 2))
        return q + y
# Inputs to the model
x1 = torch.randn(1, 2, 2)
