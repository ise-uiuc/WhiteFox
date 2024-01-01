
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.2)
 
    def forward(self, x1, x2):
        v1 = x1.matmul(x2.transpose(-2, -1))
        v2 = v1 * 0.125
        v3 = v1.softmax(dim=-1)
        v4 = self.dropout(v3)
        return v4.matmul(x2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 32, 768)
x2 = torch.randn(128, 513, 768)
