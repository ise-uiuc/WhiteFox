
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256, bias=False)
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = torch.matmul(v1, x2.transpose(-2, -1))
        scale = 5
        v3 = v2 / scale
        v4 = self.softmax(v3)
        v5 = self.dropout(v4)
        output = v5.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 64)
x2 = torch.randn(1, 64, 256)
