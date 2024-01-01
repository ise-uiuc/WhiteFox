
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=128, num_heads=16, batch_first=True)
 
    def forward(self, inputs):
        q, k, v = inputs
        return self.attn(q, k, v)

# Initializing the model
m = Model()
 
# Inputs to the model
q = k = v = torch.randn(8, 16, 128)
inputs = (q, k, v)
