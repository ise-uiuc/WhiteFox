
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(0)

    def forward(self, x1, x2, dropout_p):
        v1 = x1 @ x2.transpose(-2, -1)
        v1 = v1 / math.sqrt(v1.size(-1))
        v1 = v1 + torch.ones_like(v1) * (-100.0)
        v1 = self.softmax(v1)
        v1 = self.dropout(v1, dropout_p, True)
        output = v1 @ x2
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 6, 512)
x2 = torch.randn(8, 512, 6)
dropout_p = 0.2
