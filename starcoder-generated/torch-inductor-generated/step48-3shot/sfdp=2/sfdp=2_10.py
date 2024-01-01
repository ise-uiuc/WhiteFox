
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs for the model
query = torch.randn(5, 8, 128) # the batch size, the number of heads, and the dimension per head of the query tensor
key = torch.randn(5, 8, 128) # the batch size, the number of heads, and the dimension per head of the key tensor
value = torch.randn(6, 8, 128) # the batch size, the number of heads, and the dimension per head of the value tensor
inv_scale_factor = torch.randn(6, 8, 8) # the batch size, the number of heads, and the dimension per head of the inverse scale factor tensor
dropout_p = 0.5

