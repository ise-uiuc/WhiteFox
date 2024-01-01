
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32, bias=True)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1).relu()
        v2 = torch.matmul(x2, x3.T)
        v3 = torch.div(v2, 0.1)
        v4 = F.softmax(v3, dim=-1)
        v5 = F.dropout(v4, p=0.4, training=torch.is_grad_enabled())
        v6 = torch.matmul(v5, x2)
        return v1 + v6

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(4, 16)
x2 = torch.randn(4, 8, 8)
x3 = torch.randn(8, 8)
