
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x):
        a = torch.nn.functional.dropout(x, p=0.3)
        b = self.linear(a) + 1
        return a
# Inputs to the model
x1 = torch.randn(1, 2, 3) 
