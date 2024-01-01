
class Model(torch.nn.Module):
    def __init__(self,
                 query,
                 key,
                 value,
                 scale,
                 dropout_p):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.functional.dropout
        self.matmul1 = torch.matmul
        self.matmul2 = torch.matmul
        self.mul = torch.mul
        self.div = torch.div
  
    def forward(self, q, k, v, x3):
        qk = self.matmul1(q, k.transpose(-1, -2))
        scaled_qk = self.mul(qk, scale)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk, p=dropout_p)
        output = self.matmul2(dropout_qk, v)
        return output

# Initializing the model
query = torch.randn(1, 32, 64)
key = torch.randn(1, 32, 128)
value = torch.randn(1, 32, 128)
scale = float((1/math.sqrt(128)))
dropout_p = float(0.5)
m = Model(query, key, value, scale, dropout_p)

# input to the model
x3 = float(0.001)
