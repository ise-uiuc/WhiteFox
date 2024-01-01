
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, **kwargs):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(kwargs['inv_scale_factor'])
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=kwargs['dropout_p'])
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 50, 64)
key = torch.randn(1, 4, 100, 64)
value = torch.randn(1, 4, 100, 64)
kwargs = {'inv_scale_factor': 1,'dropout_p': 2/6}
