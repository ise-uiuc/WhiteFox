
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 10, 20)
key = torch.randn(1, 10, 30)
value = torch.randn(1, 10, 30)
