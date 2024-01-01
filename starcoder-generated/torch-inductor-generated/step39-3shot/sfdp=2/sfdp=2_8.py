
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dropout_p=0.5):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 2
        dropout_qk = torch.nn.functional.dropout(qk.div(inv_scale_factor).softmax(dim=-1), p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 3, 16)
key = torch.randn(2, 3, 20)
value = torch.randn(2, 3, 20)
dropout_p = 0.3
