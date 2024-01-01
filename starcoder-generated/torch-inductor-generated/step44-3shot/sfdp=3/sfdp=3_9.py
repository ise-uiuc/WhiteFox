
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, w1, w2):
        x3 = torch.matmul(x1, x2.transpose(-2, -1))
        x4 = x3 * w1
        x5 = x4.softmax(dim=-1)
        x6 = torch.nn.functional.dropout(x5, p=w2)
        x7 = torch.matmul(x6, w)
        return x7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(2, 5)
w1 = torch.tensor(2.0)
w2 = 0.5
