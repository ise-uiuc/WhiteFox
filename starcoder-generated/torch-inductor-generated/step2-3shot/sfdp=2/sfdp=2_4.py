
class Model(torch.nn.Module):
    def forward(self, query, key, value, p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = ( query.size(-1) ) ** 0.5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=p)
        output = dropout_qk.matmul(value,b)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 30, 24)
key = torch.randn(1, 40, 24)
value = torch.randn(1, 40, 24)
p = torch.rand()
