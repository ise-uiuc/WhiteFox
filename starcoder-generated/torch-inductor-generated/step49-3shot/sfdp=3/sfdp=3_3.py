
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
 
    def forward(self, query, key, value, mask):
        scale_factor = (key.shape[-1] ** -0.5)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = ScaledDotProductAttention(0.3)

# Inputs to the model
query = torch.randn(1, 1, 64) # (Batch size, maximum length of a query, dimension of a query)
key   = torch.randn(1, 10, 64) # (Batch size, maximum length of a key, dimension of a key)
value = torch.randn(1, 10, 64) # (Batch size, maximum length of a value, dimension of a value)
mask  = torch.rand(1, 10) < 0.5 # (Batch size, 10)
