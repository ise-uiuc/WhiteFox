
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 0.7
        v3 = self.softmax(v2)
        v4 = self.dropout(v3)
        v5 = torch.matmul(v4, x2)
        out = torch.tanh(v5)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 8, 64) # input_tensor
x2 = torch.randn(1, 8, 64, 32) # query
x3 = torch.randn(1, 8, 32, 16) # key
# x4 = torch.randn(1, 8, 16, 8) # value
