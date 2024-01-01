
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor =...
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        p =...
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()
# Inputs to the model
query = torch.randn(2, 128, 16)
key = torch.randn(2, 128, 16)
value = torch.randn(2, 128, 16)
