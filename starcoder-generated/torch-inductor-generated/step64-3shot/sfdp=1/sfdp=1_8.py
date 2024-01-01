
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor=None):
        qk = query.matmul(key.transpose(-2, -1))
        if scale_factor is not None:
            scaled_qk = qk.div(scale_factor)
        else:
            scaled_qk = qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=1.)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 8)
key = torch.randn(1, 8, 8)
value = torch.randn(1, 8, 8)
