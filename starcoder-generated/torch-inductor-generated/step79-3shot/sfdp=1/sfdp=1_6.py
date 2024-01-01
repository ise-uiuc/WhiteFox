
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = math.sqrt(query.size(-1))
        scaled_qk = qk.div(inv_scale_factor)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output


# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(12, 3, 8)
key = torch.randn(10, 3, 8)
value = torch.randn(10, 3, 8)
