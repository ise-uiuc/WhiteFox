
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = query.matmul(key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor.unsqueeze(-1).unsqueeze(-1))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
query = torch.randn(2, 5, 2, 6) # Query tensor
key = torch.randn(2, 4, 5, 6) # Key tensor
value = torch.randn(2, 4, 2, 6) # Value tensor

inv_scale_factor = torch.randn(2, 2) # Inverse scale factor tensor

dropout_p = 0.75 # Dropout probability

