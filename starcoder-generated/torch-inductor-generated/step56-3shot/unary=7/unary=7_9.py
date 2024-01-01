
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.1666666716337204
        v3 = v2 * 0.3333333432674408
        v4 = v3 * 0.5
        v5 = v4 * 0.6666666865348816
        v6 = v5 * 0.8333333134651184
        v7 = v6 * 0.5
        v8 = v7 * 0.3333333432674408
        v9 = v8 * 0.1666666716337204
        v10 = v9 + 1.0
        return v10

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(64)
