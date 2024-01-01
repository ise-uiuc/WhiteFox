
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x, y):
        x = self.linear1(x)
        x = F.dropout(x, 0.2)
        z = torch.nn.functional.relu6(x)
        y = torch.rand_like(x)
        y = y - 0.5
        z = z - y
        return z
# Input to the model
y = torch.randn(1, 2, 2)
x = torch.randn(1, 2, 2)
