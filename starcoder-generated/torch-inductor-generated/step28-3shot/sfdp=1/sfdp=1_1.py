
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, masking_key=None, masking_value=None, dropout_p=0.0):
        size = (query.size(-1), key.size(-1))
        inv_scale_factor = math.sqrt(size[1])
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)) / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = ScaledDotProductAttention()

# Inputs to the model
query = torch.randn(1, 16, 512)
key = torch.randn(1, 16, 512)
value = torch.randn(1, 16, 512)
