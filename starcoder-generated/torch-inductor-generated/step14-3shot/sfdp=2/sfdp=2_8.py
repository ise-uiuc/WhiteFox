
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query_tensor, key_tensor, value_tensor, dropout_p=0.5, inv_scale_factor=1. / math.sqrt(8)):
        qk = torch.matmul(query_tensor, key_tensor.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value_tensor)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 64)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
