
class Attention(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p):
        super().__init__()
        self.matmul1 = torch.nn.MatMul(query.size(-1), key.size(-1))
        self.matmul2 = torch.nn.MatMul(value.size(-1), key.size(-1))
        self.softmax = torch.nn.Softmax(dim=key.dim() - 1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = self.matmul1(query, key)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = self.matmul2(dropout_qk, value)
        return output

# Initializing the model
attn = Attention(query, key, value, dropout_p=0.5)

# Inputs to the model
query = torch.randn(1, 5, 3, 64)
key = torch.randn(1, 5, 4, 64)
value = torch.randn(1, 5, 4, 64)
inv_scale_factor = torch.randn(1, 5, 1, 1)
