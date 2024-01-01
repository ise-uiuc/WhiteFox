
class Model(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = dropout(softmax_qk, p=dropout_p, training=self.training)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model(heads)

# Inputs to the model
query = torch.randn(1, 32, 128)
key = torch.randn(1, 32, 256)
value = torch.randn(1, 32, 256)
scale_factor = 1/sqrt(128)
dropout_p = 0.5
