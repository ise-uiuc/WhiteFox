
class Model(torch.nn.Module):
    def __init__(self, dimension, num_heads, dropout_p=0.1):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(dimension, num_heads)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, k, q, v, mask):
        v, v_weight, v_pos = v
        outputs = self.mha(q, k, v, None, None, mask)[0]
        return self.dropout(outputs)

# Initializing the model 
m = Model(dimension, num_heads)

# Inputs to the model
k = torch.randn(8, sequence_length, dimension)
q = torch.randn(8, num_heads, sequence_length, dimension // num_heads)
v = torch.randn(8, sequence_length + 1, dimension)
mask = torch.ones([8, 1, 1, sequence_length])
