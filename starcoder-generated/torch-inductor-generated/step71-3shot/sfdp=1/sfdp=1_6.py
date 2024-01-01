
class Model(torch.nn.Module):
    def __init__(self, shape, dropout_p):
        super().__init__()
        self.shape = shape
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
shape = [12, 32, 10]
m = Model(shape, 0.5)

# Inputs to the model
query = torch.randn(shape)
key = torch.randn(shape)
value = torch.randn(shape)
inv_scale_factor = torch.randn(shape[1])
