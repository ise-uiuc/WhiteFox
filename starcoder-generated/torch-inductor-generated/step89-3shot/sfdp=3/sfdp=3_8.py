
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mul = torch.nn.Mul()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout()
        self.matmul = torch.nn.MatMul(transpose_b=True)
    
    def forward(query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = self.mul((scale_factor, qk))
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk, p=dropout_p)
        output = self.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, 8, 100)
key = torch.randn(16, 32, 100)
value = torch.randn(16, 32, 100)
scale_factor = torch.randn((1, 1, 100))
dropout_p = 0.5
