
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 128
        self.num_heads = 12

    def forward(self, q, k, v):
        inv_scale_factor = (self.embed_dim // self.num_heads) ** -0.5
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.4)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, 16, 128)
key = torch.randn(16, 16, 128)
value = torch.randn(16, 16, 128)
