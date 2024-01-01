
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
 
    def forward(self, x1, x2):
        w = self.linear(x1)
        v = w.matmul(x2.transpose(-2, -1))
        v1 = v.div(256)
        v2 = torch.nn.functional.softmax(v1, dim=-1)
        v3 = torch.nn.functional.dropout(v2, 0.5, training=True)
        v4 = v3.matmul(x2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 8, 64)
