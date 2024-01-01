
class Model(torch.nn.Module):
    def __init__(self, query,key,value):
        super().__init__()
        self.query = Parameter(query.data)
        self.key = Parameter(key.data)
        self.value = Parameter(value.data)
 
    def forward(self, qk):
        inv_scale_factor = 1.0 / math.sqrt(self.query.size(-1))
        inv_scale_factor = to_tensor([inv_scale_factor])
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=(0.5))
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
query = torch.randn(1, 4, 2)
key = torch.randn(1, 5, 2)
value = torch.randn(1, 5, 7)
m = Model(query, key, value)

# Inputs to the model
qk = torch.randn(1, 4, 5)
