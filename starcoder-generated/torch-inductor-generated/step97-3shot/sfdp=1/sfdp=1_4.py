
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_qk = torch.nn.Dropout(0.1)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale = math.sqrt(query.size(-1))
        scaled_qk = qk.div(scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_qk(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 96, self_attn_width)
key = torch.randn(1, 3, self_attn_width, 64)
value = torch.randn(1, 3, 64, 128)
