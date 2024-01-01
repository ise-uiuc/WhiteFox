
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mul = torch.nn.Parameter(torch.tensor(10.), requires_grad=True)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        v = torch.nn.functional.dropout(qk.softmax(dim=-1).mul(self.mul), p=0.5)
        return torch.matmul(v, x2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 5, 10)
x2 = torch.randn(2, 10, 15)
