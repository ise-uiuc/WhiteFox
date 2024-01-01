
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(0.1)
    def forward(self, input):
        x = self.linear1(input)
        x = self.dropout(x)
        y = torch.rand_like(x)
        z = torch.rand_like(x)
        return torch.mul(y, z)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
