
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor=None, dropout_p=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        if inv_scale_factor is not None:
            scaled_qk = qk.div(inv_scale_factor)
        else:
            scaled_qk = qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        if dropout_p is not None:
            dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)            
        else:
            dropout_qk = softmax_qk
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1,8,64)
key = torch.randn(1,8,64)
value = torch.randn(1,8,64)
dropout_p = torch.tensor(0.5)
inv_scale_factor = torch.tensor(math.sqrt(64))
