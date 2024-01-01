
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 10)
    def forward(self, x):
        r1 = torch.nn.functional.dropout(self.linear(x), p=0.3)
        return r1
# Inputs to the model
x = torch.randn(1, 100)
