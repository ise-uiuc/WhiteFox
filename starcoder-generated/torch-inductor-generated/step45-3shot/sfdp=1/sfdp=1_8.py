
class SingleHeadAttention(torch.nn.Module):
    def __init__(self, embedding_size, dropout_p=0.1):
        super().__init__()
        assert embedding_size % 2 == 0, "Embedding size should be an even value."
        self.embedding_size = embedding_size
        self.head_dim = embedding_size // 2
        w = torch.empty(self.head_dim, self.head_dim)
        self.w_q = torch.nn.Linear(self.head_dim, self.head_dim)
        self.w_kv = torch.nn.Linear(self.head_dim, 2 * self.head_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        
        torch.nn.init.xavier_uniform_(self.w_q.weight)
        torch.nn.init.xavier_uniform_(self.w_kv.weight)
        torch.nn.init.zeros_(self.w_q.bias)
        torch.nn.init.zeros_(self.w_kv.bias)

    def forward(self, query, key, value, mask=None):
        query_length = query.size(-2)
        key_length = key.size(-2)
    
        q = self.w_q(query)
        kv = self.w_kv(key)
        kv = torch.reshape(kv, [key_length, query_length, self.head_dim, 2])
        k = kv[:, :, :, 0]
        v = kv[:, :, :, 1]

        dot_product = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.sqrt(torch.Tensor([self.head_dim])).to(
            dot_product.device)
        scaled_dot_product = dot_product.divide(inv_scale_factor)
        scaled_dot_product = scaled_dot_product.masked_fill(
            mask.unsqueeze(1) == 0, float('-inf'))
        softmax_dot_product = scaled_dot_product.softmax(-1)

        self.dropout.to(softmax_dot_product.device)
        dropout_output = self.dropout(softmax_dot_product)

        output = torch.matmul(dropout_output, v)
        return output

class Model(torch.nn.Module):
    def __init__(self, embedding_size, num_heads, dropout_p=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
 
        self.head = SingleHeadAttention(embedding_size, dropout_p)
        self.transformer = torch.nn.Transformer(d_model=embedding_size, nhead=num_heads)
        self.fc = torch.nn.Linear(embedding_size, embedding_size)
 
    def forward(self, x):
        x = torch.nn.functional.pad(x, [0, 0, self.num_heads, self.num_heads])
        x = self.transformer(x, src_key_padding_mask=get_mask(x))
        x = self.head(x, x, x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.fc(x)
 
        return x

# Initializing the model
m = Model(200, 10)

# Inputs to the model
x1 = torch.randn(20, 32, 200)
