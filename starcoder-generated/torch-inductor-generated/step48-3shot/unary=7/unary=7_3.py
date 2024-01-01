
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2048, 1024)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.clamp(v1 + 3, min=0, max=6)
        v3 = v2 * 0.16666666666666666
        v4 = v2 * 0.3333333333333333
        v5 = v2 * 0.5
        v6 = v2 * 0.6666666666666666
        v7 = v2 * 0.8333333333333334
        v8 = v2
        return [v7, v6, v5, v4, v3, v2, v1]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2048)
