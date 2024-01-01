
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(n_head, d_model, dropout_p)
 
    def forward(self, query, key, value):
        vq, vv = self.attention(query, key, value)
        return vq

# Initializing the model
n_head = 3
d_model = 10
dropout_p = 0.2
query = torch.randn(10, 30, d_model)
key = torch.randn(20, 30, d_model)
value = torch.randn(20, 30, d_model)
