
class Model(torch.nn.Module):
    def __init__(self, dim_model=8, num_attention_heads=8, dropout_p=0.2):
        super().__init__()
        self.dot_product_attention = BaseScaledDotProductAttention(dropout_p=dropout_p, dim_model=dim_model, num_heads=num_attention_heads)
 
    def forward(self, k, q, v, mask=None):
        output = self.dot_product_attention(q, k, v, mask=mask)
        return output, None

# Initializing the model
m = Model(dim_model=8, num_attention_heads=8, dropout_p=None)

# Inputs to the model
__k__ = torch.randn(1, 32, 8)
__q__ = torch.randn(1, 32, 8)
__v__ = torch.randn(1, 32, 8)
__mask__ = torch.tensor([[0, 1, 0]])
