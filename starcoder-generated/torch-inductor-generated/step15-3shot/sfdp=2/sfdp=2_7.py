
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super(Model, self).__init__()
        self.num_heads = num_heads

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1.0 / math.sqrt(math.sqrt(self.num_heads))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_pk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(num_heads=64)

# Generate random data
shape = (16, 128)
query = torch.randn(shape)
key = torch.randn(shape)
value = torch.randn(shape)
