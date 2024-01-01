
class Model(torch.nn.Module):
    def __init__(self, dim, drop_rate):
        super().__init__()
        self.dropout_p = drop_rate
        self.inv_scale_factor = dim ** -0.5
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p).matmul(value)
        return output

# Initializing the model
model = Model(128, 0.1)

# Inputs to the model
query = torch.randn(1, 128, 15)
key = torch.randn(1, 128, 20)
value = torch.randn(1, 128, 20)
