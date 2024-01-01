
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.ones(10,16))
        self.value = torch.nn.Parameter(torch.ones(10,16))
 
    def forward(self, x1):
        x2 = torch.matmul(x1, self.key.transpose(-2, -1))
        x3 = x2.div(2.0 ** 0.5)
        x4 = torch.nn.functional.softmax(x3, dim=-1)
        x5 = torch.nn.functional.dropout(x4, p=0.1)
        x6 = torch.matmul(x5, self.value)
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 16)
