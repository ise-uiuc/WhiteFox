
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.rand_like(torch.rand((1, 1, 1)))
    def forward(self, x):
        x = torch.nn.functional.dropout(x)
        x = torch.randint_like(x, 0, 10)
        x = torch.tensor([-10, -7, 2, 7, -7, -2, -9, 0, -2, 2])
        x.unsqueeze_(-1)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = F.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(x)
        return x
# Inputs to the model
x1 = torch.randn(1)
