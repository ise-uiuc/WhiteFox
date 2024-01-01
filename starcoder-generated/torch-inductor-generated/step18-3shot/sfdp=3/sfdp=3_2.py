
class Model(torch.nn.Module):
    def __init__(self, attention_dim, num_attn_heads, dropout_p):
        super().__init__()
        w_init_range = 0.1
        self.projection_dim = attention_dim * num_attn_heads
        self.query = torch.nn.Linear(self.projection_dim, self.projection_dim, bias=False)
        self.key = torch.nn.Linear(self.projection_dim, self.projection_dim, bias=False)
        self.value = torch.nn.Linear(self.projection_dim, self.projection_dim, bias=False)
        self.output = torch.nn.Linear(self.projection_dim, self.projection_dim, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
        torch.nn.init.normal_(self.key.weight, std=w_init_range)
        torch.nn.init.normal_(self.query.weight, std=w_init_range)
        torch.nn.init.normal_(self.value.weight, std=w_init_range)
        torch.nn.init.normal_(self.output.weight, std=w_init_range)
 
    def forward(self, query, key, value, mask):
        q = self.query(query).view(-1, self.num_attn_heads, self.attention_dim).permute(1,0,2)
        k = self.key(key).view(-1, self.num_attn_heads, self.attention_dim).permute(1,0,2)
        v = self.value(value).view(-1, self.num_attn_heads, self.attention_dim).permute(1,0,2)
        q = kq = torch.nn.functional.dropout(q, p=dropout_p, training=self.training)
        v = kv = torch.nn.functional.dropout(v, p=dropout_p, training=self.training)
        scores_per_head = torch.matmul(q, k.transpose(-2, -1)).mul(self.scale_factor)
        scores_per_head = self.softmax(scores_per_head)
        output_per_head = torch.matmul(scores_per_head, kv).permute(1,0,2)
        final_output = self.output(output_per_head.flatten(1))
        return final_output


# Initializing the model
m = Model('./model')

# Inputs to the model
query = torch.randn(1, 1280)
key = torch.randn(2, 1280)
value = torch.randn(2, 1280)
mask = torch.zeros(1,2).long()
