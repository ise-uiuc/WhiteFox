
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 8
        self.head_dim = 64
        self.kqv_dim = self.num_heads * self.head_dim
        embed_dim = 768
        self.qkv = torch.nn.Linear(embed_dim, 4 * self.kqv_dim)
        self.proj_out = torch.nn.Linear(self.kqv_dim, embed_dim)
        self.dropout_p = 0.1
        self.inv_scale_factor = torch.sqrt(torch.FloatTensor([0.25 / self.head_dim]))
        self.query = torch.nn.Parameter(torch.rand([1, 1, embed_dim]))
        self.key = torch.nn.Parameter(torch.rand([1, 1, embed_dim]))
        self.value = torch.nn.Parameter(torch.rand([1, 1, embed_dim]))
 
    def forward(self):
        qkv = self.qkv(self.query)
        query, key, value = torch.chunk(qkv, chunks=self.num_heads * 3, dim=-1)
        q, k, v = torch.chunk(query, chunks=self.num_heads, dim=-1)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        qk = torch.matmul(q, k)
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        result = self.proj_out(output.transpose(-2, -1).contiguous()).squeeze(-2)
        return result, query, key, value

# Initializing the model
model = Model()

x = torch.ones(64, 768)
y, q, k, v = model(x)
