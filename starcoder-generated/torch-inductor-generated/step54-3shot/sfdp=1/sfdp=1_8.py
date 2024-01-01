
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
 
    def forward(self, x1, x2):
        t1 = torch.matmul(x1, x2.transpose(-2, -1))
        t2 = t1.div(10)
        t3 = torch.nn.functional.softmax(t2, dim=-1)
        t4 = torch.nn.functional.dropout(t3, p=0.3)
        t5 = torch.matmul(t4, x2)
        return t5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
