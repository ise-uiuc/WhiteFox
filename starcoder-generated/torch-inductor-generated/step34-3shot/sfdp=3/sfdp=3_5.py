
embed_dim = 512
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
model = Model()

# Inputs to the model
query = torch.randn(1, 2, 3, embed_dim)
key = torch.randn(1, 2, 5, embed_dim)
value = torch.randn(1, 2, 5, embed_dim)
scale_factor = torch.Tensor([math.sqrt(float(embed_dim))])
