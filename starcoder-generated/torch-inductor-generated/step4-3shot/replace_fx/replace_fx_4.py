
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x):
        v1 = torch.nn.functional.dropout(x)
        v2 = torch.rand_like(x)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2)
