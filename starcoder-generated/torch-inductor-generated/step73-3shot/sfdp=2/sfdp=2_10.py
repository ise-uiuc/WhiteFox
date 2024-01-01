
class Model(torch.nn.Module):
    def __init__(self, query, key, value, inv_scale_factor, dropout_p):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 256)
value = torch.randn(1, 8, 256, 64)
inv_scale_factor = torch.randn(1, 8, 1, 1)
dropout_p = 0.25
