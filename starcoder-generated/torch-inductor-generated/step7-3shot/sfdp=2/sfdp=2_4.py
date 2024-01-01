
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(4, 4))
        self.key = torch.nn.Parameter(torch.randn(4, 4))
        self.value = torch.nn.Parameter(torch.randn(4, 4))
 
    def forward(query, key, value, dropout_p, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(2, 4)
key = <KEY>)
value = torch.randn(2, 4)
dropout_p = 0.5
inv_scale_factor = math.sqrt((query.shape[1] - 1) * 2)
