
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def scaled_dot_product_attention(self, query, key, value, attn_mask):
        dimension_batch_size = query.shape[0]
        scaled_product = torch.matmul(query, key.T) / math.sqrt(query.size(-1))
        scaled_product = scaled_product + attn_mask
        weight = torch.nn.Softmax(dim = -1)(scaled_product)
        result = torch.matmul(weight, value)
        return result
 
    def forward(self, q, k, v, attn_mask):
        v = self.scaled_dot_product_attention(q, k, v, attn_mask)
        return v

# Initializing the model
m = Model()
m_ref = Model()

# Inputs to the model
q = torch.randn(10, 3, 10, 10)
k = torch.randn(10, 4, 10, 10)
v = torch.randn(10, 5, 10, 10)
attn_mask = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
