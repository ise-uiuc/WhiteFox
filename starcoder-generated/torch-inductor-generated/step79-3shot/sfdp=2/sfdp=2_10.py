
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor=None, dropout_p=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        if inv_scale_factor is not None:
            qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        if dropout_p is not None:
            softmax_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = softmax_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 32, 64)
key = torch.randn(1, 32, 64)
value = torch.randn(1, 32, 64)
inv_scale_factor = torch.tensor(1.0)
dropout_p = torch.tensor(0.5)
