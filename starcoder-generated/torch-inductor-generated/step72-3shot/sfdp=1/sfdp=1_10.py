
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
model = Model( )

# Initializing the inputs for the model (query, key, value, inv_scale_factor and dropout_p tensors)
query = torch.randn(2, 4, 1)
key = torch.randn(2, 1, 20)
value = torch.randn(2, 1, 20)
inv_scale_factor = torch.arange(1.0, 21.0).view(1, 1, 20)
dropout_p = 0.5
