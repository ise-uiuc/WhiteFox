
class Model(torch.nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
 
    def forward(self, query, key, value, dropout_p=0.5, inv_scale_factor=1.4142135623730951):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 16, 256)
key = torch.randn(3, 32, 256)
value = torch.randn(3, 32, 128)
