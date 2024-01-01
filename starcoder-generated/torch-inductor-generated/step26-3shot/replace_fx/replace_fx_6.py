
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1)
        x3 = torch.rand_like(x2, requires_grad=True)
        return x3
# Inputs to the model
x1 = torch.rand(10, 10)
