
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
m = Model().eval()

# Inputs to the model
query = torch.randn(2, 5, 512, 64)
key = torch.randn(2, 5, 64, 64)
value = torch.randn(2, 5, 64, 64)
inv_scale_factor = torch.randn(2, 5, 1, 1)
dropout_p = torch.randn([])
