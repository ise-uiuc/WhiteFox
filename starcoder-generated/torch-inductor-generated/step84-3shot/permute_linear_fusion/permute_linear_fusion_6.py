
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.softmax = torch.nn.Softmax(dim=0)
        self.t2 = torch.randn(2)
        self.softmax = torch.nn.Softmax(dim=0)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, None, self.t2)
        v3 = self.softmax(v2)
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, 2)
