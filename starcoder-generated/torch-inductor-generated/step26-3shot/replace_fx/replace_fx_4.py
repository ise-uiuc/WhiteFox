
class model(torch.nn.Module):
     def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, x):
        x1 = self.dropout(x)
        x2 = torch.rand_like(x1, requires_grad=True)
        return x1
# Inputs to the model
x1 = torch.randn(10, 4)
