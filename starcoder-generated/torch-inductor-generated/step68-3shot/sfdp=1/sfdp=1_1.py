
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def calc_scaled_dot_prod(self, q, k, i):
        v = torch.matmul(q, k.transpose(-2, -1))
        v = v.div(i)
        return v
 
    def forward(self, query, key, value, scale_factor, dropout_p, mask):
        scaled_dot_prod = self.calc_scaled_dot_prod(query, key, scale_factor)
        softmax_dot_prod = scaled_dot_prod.softmax(dim=-1)
        dropout_dot_prod = torch.nn.functional.dropout(softmax_dot_prod, p=dropout_p, training=self.training)
 
        if mask is not None:
            dropout_dot_prod = dropout_dot_prod.masked_fill(mask == 0, -1e9)
 
        res = torch.matmul(dropout_dot_prod, value)
 
        return res

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 1, 1, 16)
key = torch.randn(4, 1, 16, 1)
value = torch.randn(4, 1, 16, 8)
