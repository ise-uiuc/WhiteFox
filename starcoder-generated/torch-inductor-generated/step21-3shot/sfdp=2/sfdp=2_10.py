
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, value, query, dropout_p=0.1):
        qk = torch.matmul(query, value.transpose(-2, -1))
        inv_scale_factor = torch.sqrt(torch.tensor(query.size(-1)).float())
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 16, 128)
value = torch.randn(1, 64, 16)
query = torch.randn(1, 8, 128)
