
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = x2.permute(1, 0, 2)
        v4 = x2.permute(2, 1, 0)
        vs = [
            torch.bmm(v1, self.linear.weight),
            torch.bmm(v1, v1),
            torch.matmul(v3, self.linear.weight),
            torch.matmul(v3, v4),
        ]
        return vs

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
