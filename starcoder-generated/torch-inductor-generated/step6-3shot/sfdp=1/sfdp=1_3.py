
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, dropout_p=0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1. / np.sqrt(math.sqrt(qk.size(-1)))
        dropout_qk = torch.nn.functional.dropout(
            qk.div(inv_scale_factor).softmax(dim=-1),
            p=dropout_p
        )
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 8, 64, 64)
value = torch.randn(1, 8, 64, 64)
