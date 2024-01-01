
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self,  y, x):
        a=self.linear(x)
        return a+torch.nn.functional.dropout(y, p=0.5)
# Inputs to the model
x1 = torch.randn(1, 2, 4)
y = torch.exp(torch.randn(1, 1, 1))
