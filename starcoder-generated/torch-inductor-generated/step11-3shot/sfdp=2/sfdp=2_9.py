
class SelfAttentionOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, dropout_p=0, inv_scale_factor=1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output, qk, softmax_qk, scaled_qk

attention = SelfAttentionOutput()

# Inputs to the model
query = torch.randn(4, 2, 8)
key = torch.randn(4, 2, 8)
value = torch.randn(4, 2, 4)
__output__, _, _, __qk__ = attention(query, key, value, dropout_p=0.1, inv_scale_factor=2)

__all__ = [
    "m"
]
