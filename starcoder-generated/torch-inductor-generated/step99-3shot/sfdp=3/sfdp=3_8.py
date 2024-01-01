
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1_1 = torch.nn.MatMul()
        self.matmul2_1 = torch.nn.MatMul()
        self.matmul3_1 = torch.nn.MatMul()
        self.dropout = torch.nn.Dropout()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = self.matmul1_1(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk, p=dropout_p)
        output = self.matmul3_1(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 3, 16)
key = torch.randn(4, 3, 16)
value = torch.randn(4, 3, 16)
p1 = torch.tensor(0.05)
