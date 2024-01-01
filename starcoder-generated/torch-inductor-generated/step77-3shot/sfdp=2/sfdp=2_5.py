
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, queries, keys, values, inverse_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dropout_p=0.1)

# Inputs to the model
q = torch.randn(10, 5, 1024)
k = torch.randn(10, 5, 128)
v = torch.randn(10, 5, 128)
inv_scale_factor = torch.rand(10, 5, 128)
