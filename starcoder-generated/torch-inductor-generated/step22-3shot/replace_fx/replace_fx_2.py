
class Model(torch.nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, d)
    def forward(self, x1):
        x2 = self.linear1(x1)
        x3 = torch.nn.functional.dropout(x2)
        x4 = self.linear2(x3)
        x5 = torch.nn.functional.gelu(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2)
