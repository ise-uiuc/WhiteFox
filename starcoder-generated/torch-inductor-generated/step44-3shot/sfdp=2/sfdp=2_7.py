
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
 
    def forward(self, query, key, value, training):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * (1. / sqrt(self.num_heads))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(num_heads=12)

# Inputs to the model
query = torch.randn(1, 12, 128, 64)
key = torch.randn(1, 12, 128, 64)
value = torch.randn(1, 12, 128, 64)
training = True
