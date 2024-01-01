
class DotProductAttention(nn.Module):
    def __init__(self, dropout_p, scale):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale = scale
 
    def forward(self, query, key, value, mask=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = torch.matmul(dropout_qk, value)
 
        return output

# Initializing the model
dropout_p = 0.5
scale = math.sqrt(dk)
m = DotProductAttention(dropout_p, scale)

# Inputs to the model
query = torch.randn(1, 4, 10)
key = torch.randn(2, 4, 10)
value = torch.randn(2, 4, 20)
