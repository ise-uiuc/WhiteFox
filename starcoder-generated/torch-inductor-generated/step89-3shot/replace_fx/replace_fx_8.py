
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.dropout1 = torch.nn.Dropout(0.5)
    def forward(self, x, y):
        z = self.linear1(x) + y
        x = self.dropout1(z)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2)
