
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_dropout = torch.nn.Dropout2d()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.attention_dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 256, 384)
key = torch.randn(1, 256, 384)
value = torch.randn(1, 256, 384)
inv_scale_factor = 10.0 ** 3
dropout_p = 0.5
