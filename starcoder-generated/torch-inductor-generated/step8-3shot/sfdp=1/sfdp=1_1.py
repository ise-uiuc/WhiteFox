
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_scale_factor = 1.0
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.attention_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 8)
key = torch.randn(1, 8, 16)
value = torch.randn(1, 16, 64)
dropout_p = 0.2
