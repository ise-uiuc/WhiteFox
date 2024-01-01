
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2000, 800)
        self.linear2 = torch.nn.Linear(800, 400)
        self.linear3 = torch.nn.Linear(400, 200)
        self.linear4 = torch.nn.Linear(200, 3)
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = torch.tanh(v1)
        v3 = self.linear2(v2)
        v4 = torch.tanh(v3)
        v5 = self.linear3(v4)
        v6 = torch.tanh(v5)
        v7 = self.linear4(v6)
        v8 = torch.tanh(v7)
        return v8
# Inputs to the model
x = torch.randn(1, 2000)
