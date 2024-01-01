
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.dropout = torch.nn.Dropout()
        self.gelu = torch.nn.GELU()
    def forward(self, x):
        a = self.linear(x)
        b = torch.log_softmax(a, dim=-1)
        a1 = self.gelu(a)
        b0 = self.dropout(b)
        c = self.gelu(a1)
        return b0
# Inputs to the model
x1 = torch.randn(1, 2, 4)
