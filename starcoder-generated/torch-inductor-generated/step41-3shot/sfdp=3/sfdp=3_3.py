
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
    
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = (self.num_heads*1.0) ** -0.25
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(num_heads=4)

# Inputs to the model
query = torch.randn(1, 4, 8)
key = torch.randn(1, 16, 8)
value = torch.randn(1, 16, 8)
dropout_p = 0.8
