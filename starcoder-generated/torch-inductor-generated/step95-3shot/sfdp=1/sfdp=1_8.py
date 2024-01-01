
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(2, 4, 2))
        self.key = torch.nn.Parameter(torch.randn(2, 3, 2))
        self.value = torch.nn.Parameter(torch.randn(2, 3, 2))
        self.softmax_dropout = torch.nn.functional.softmax + torch.nn.functional.dropout
 
    def forward(self, x1):
        v1 = torch.matmul(x1, self.query.transpose(-2, -1))
        v2 = v1.div(0.1)
        v3 = self.softmax_dropout(v2, p=0.5)
        v4 = torch.matmul(v3, self.value)
        return v4

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 2, 4)
