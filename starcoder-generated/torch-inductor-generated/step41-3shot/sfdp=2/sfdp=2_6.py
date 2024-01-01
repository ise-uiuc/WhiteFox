
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p):
        super().__init__()
        self.dim = dim
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, training=True):
        inv_scale_factor = 1 / math.sqrt(self.dim)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p, training=training)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
dim, dropout_p = 10, 0.5
m = Model(dim, dropout_p)

# Inputs to the model
query = torch.randn(2, 7, dim)
key = torch.randn(2, 6, dim)
value = torch.randn(2, 6, dim)
