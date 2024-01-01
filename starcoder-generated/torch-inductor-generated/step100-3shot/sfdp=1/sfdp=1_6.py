
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
m = Model1()

# Inputs to the model
query = torch.randn(4, 8, 4, 4)
key = torch.randn(4, 16, 2, 2)
value = torch.randn(4, 16, 2, 2)
inv_scale_factor = torch.randn(16)
dropout_p = torch.tensor(0.1)
