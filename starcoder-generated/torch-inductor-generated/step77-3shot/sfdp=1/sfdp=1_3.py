
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, inv_scale_factor, dropout_p, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
 
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
key = torch.randn(2, 2, 4, 4)
query = torch.randn(2, 2, 4, 4)
inv_scale_factor = torch.tensor([1.0])
dropout_p = torch.tensor([0.0])
value = torch.randn(2, 2, 4, 4)
