
class Model(torch.nn.Module):

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = self.model_dim // self.num_heads

    def forward(self, query, key, value, mask=None):
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5

        if mask is not None:
            scaled_dot_product.masked_fill_(mask, -float("inf"))

        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)

        return output

# Number of heads
num_heads = 32

# Initializing the model
m = Model(num_heads)

# Inputs to the model
query = torch.randn((1, num_heads, query_length, model_dim))
key = torch.randn((1, num_heads, key_length, model_dim))
value = torch.randn((1, num_heads, value_length, model_dim))
mask = torch.randn((1, 1, key_length, value_length)).to(torch.bool)
