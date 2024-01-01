
class Model(torch.nn.Module):
    num_heads = 16
    key_size = 128
    query_size = 128
    value_size = 128
    dropout = 0.5
    activation_function = torch.nn.functional.gelu
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Embedding(128, key_size)
        self.query = torch.nn.Embedding(128, query_size)
        self.value = torch.nn.Embedding(128, value_size)

    def scaled_dot_product_attention(self, query, key, value):
        attn = torch.matmul(query, key.transpose(-2, -1))
        attn = attn.div(math.sqrt(key.shape[-1]))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = torch.nn.functional.dropout(attn, p=self.dropout)
        attn = attn.matmul(value)
        return attn
 
    def forward(self, query_input, key_input):
        key = self.key(key_input)
        query = self.query(query_input)
        value = self.value(key_input)
        output = self.scaled_dot_product_attention(query, key, value)
        return output

The dimension of key and value should be the same as `key_size` and `value_size` in the model, instead of the actual shape of key and value.

# Initializing the model
m = Model()

# Input to the model
query_input = torch.randint(0, 128, (2, 16))
key_input = torch.randint(0, 128, (2, 164))
