
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, value, key=None, inv_scale_factor=0.5, dropout_p=0.5, mask=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(5, 64, 1024)
key = torch.randn(5, 64, 2048)
value = torch.randn(5, 64, 2048)
