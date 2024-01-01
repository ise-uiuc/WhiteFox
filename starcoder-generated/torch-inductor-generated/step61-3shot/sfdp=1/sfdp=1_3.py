
class Model(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)

# Initializing the model
m = Model(query, key, value)

# Inputs to the model
query = torch.randn(1, 8, 64, 64)
key = torch.randn(1, 32, 8, 64)
value = torch.randn(1, 8, 64, 64)
