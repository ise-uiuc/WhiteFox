
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, scale_factor=1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = ScaledDotProductAttention()

# Inputs to the model
query = torch.randn(1, 8, 24, 24)
key = torch.randn(1, 8, 31, 31)
value = torch.randn(1, 8, 31, 31)
scale_factor = 31.234
