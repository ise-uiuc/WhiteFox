
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8, bias=True)
        self.linear2 = torch.nn.Linear(8, 5, bias=False)
        self.linear3 = torch.nn.Linear(5, 5, bias=False)
        self.linear4 = torch.nn.Linear(5, 3, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.sigmoid(self.linear1(x1))
        v2 = torch.nn.functional.elu(self.linear2(v1))
        v3 = torch.sigmoid(self.linear3(v2))
        v4 = torch.nn.functional.elu(self.linear4(v3))
        return v4
# Inputs to the model
x1 = torch.randn(1, 3)
