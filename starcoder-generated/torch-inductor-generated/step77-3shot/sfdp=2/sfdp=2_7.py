
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
 
    def forward(self, x1, x2):
        x3 = torch.matmul(x1, x2.transpose(-2, -1))
        x4 = x3.div(3.0)
        x5 = torch.nn.functional.softmax(x4, dim=-1)
        x6 = self.dropout(x5)
        x7 = torch.matmul(x6, x2)
        return x7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
