
class Model(torch.nn.Module):
    def __init__(self, d_model=512, dropout=0.0):
        super().__init__()
        self.dropout = dropout
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1.0 / math.sqrt(math.sqrt(query.shape[-1]))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
model = Transformer(d_model=512, dropout=0.0)

# Inputs to the model
query = torch.randn(1, 64, 512)
key = torch.randn(1, 64, 512)
value = torch.randn(1, 64, 512)
