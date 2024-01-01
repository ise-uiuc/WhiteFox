
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)

        v3 = torch.max(v2, dim=-1)[1]
        v4 = self.linear(v1)

        x = torch.tensor([[[1, 1], [1.2, 1], [3, 3]]])

        w1 = torch.nn.functional.softmax(x, dim=-1)
        w2 = torch.nn.functional.sigmoid(x)
        w3 = torch.nn.functional.softplus(x)
        z = w1 + w2 + w3
        return self.linear(z)
# Inputs to the model
x1 = torch.randn(1, 3, 2)
