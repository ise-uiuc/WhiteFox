
class Model(torch.nn.Module):
    def __init__(self, dropout_p, query, key, value, inv_scale_factor):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the input tensors
dropout_p = 0.1
query = torch.randn(1, 29, 256)
key = torch.randn(1, 58, 256)
value = torch.randn(1, 58, 128)
inv_scale_factor = 1/(288**0.5)

# Initializing the model
m = Model(dropout_p, query, key, value, inv_scale_factor)

# Input to the model
x1 = torch.randn(1, 32, 256)
