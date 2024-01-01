
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(x3)
        v3 = torch.nn.functional.dropout(v2.softmax(dim=-1), p=x4)
        res = v3.matmul(x)
        return res

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 30, 8)
x2 = torch.randn(11, 32, 7)
x3 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
x4 = torch.tensor(0.5528576551433563)
