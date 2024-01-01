
class Model(torch.nn.Module):
    def __init__(self, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = 0.2
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = (key.size(-1) ** -0.5)
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 256)
key = torch.randn(1, 4, 512)
value = torch.randn(1, 4, 512)
