
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 4, 96)
key = torch.randn(3, 8, 192)
value = torch.randn(3, 8, 192)
__inv_scale_factor__ = 1.0
__dropout_p__ = 0.00B55e-2B141
