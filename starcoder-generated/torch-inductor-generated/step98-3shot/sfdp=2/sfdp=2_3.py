
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 6, 3)
key = torch.randn(4, 4, 3)
value = torch.randn(4, 4, 3)
inv_scale_factor = torch.tensor([1, 1, 1,
                                   2, 2, 2,
                                   3, 3, 3])
dropout_p = 0.2
