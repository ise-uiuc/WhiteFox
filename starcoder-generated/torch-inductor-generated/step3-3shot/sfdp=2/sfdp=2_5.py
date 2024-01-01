
class Model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.query = torch.nn.Parameter(torch.rand((1, p1)), requires_grad=True)
        self.key = torch.nn.Parameter(torch.rand((1, p1)), requires_grad=True)
        self.dropout_p = p2

    def forward(self, x1):
        v1 = torch.matmul(self.query, self.key.transpose(int(-2), int(-1)))
        v2 = v1.div(self.p1)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, self.dropout_p)
        output = v4.matmul(self.value)
        return output

# Initializing the module
query = torch.nn.Parameter(torch.rand((1, p1)), requires_grad=True)
key = torch.nn.Parameter(torch.rand((1, p1)), requires_grad=True)
value = torch.nn.Parameter(torch.rand((1, p1)), requires_grad=True)
# Inputs to the module
x1 = torch.randn(1, 1)
