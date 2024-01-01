
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.dropout1 = torch.nn.Dropout(0.2)
    def forward(self, x):
        y = self.linear1(x)
        x = F.dropout(y, p=0.2)
        y = torch.nn.functional.dropout(x, p=0.8)
        y = y ** 0.2
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
