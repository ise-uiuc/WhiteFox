
class Model(torch.nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_head
        self.attention_head_size = self.embed_dim // self.num_heads
        self.all_head_size = self.num_heads * self.attention_head_size
        self.query_weight = torch.nn.Linear(1, 1, bias=True)
        self.key_weight = torch.nn.Linear(1, 1, bias=True)
        self.value_weight = torch.nn.Linear(1, 1, bias=True)
        self.out_weight = torch.nn.Linear(1, 1, bias=True)
 
    def _transpose_for_scores(self, x, in_len):
        x = x.view(in_len, self.num_heads, self.attention_head_size)
        return x.permute(1, 0, 2).reshape(-1, in_len, self.all_head_size)
 
    def forward(self, query, key, value, input_tensor, input2_tensor, p=0.0):
        qt = self.query_weight(query)
        kt = self.key_weight(key)
        vt = self.value_weight(value)
        in_len = input_tensor.shape[2]
        q = self._transpose_for_scores(qt, in_len)
        k = self._transpose_for_scores(kt, in_len)
        v = self._transpose_for_scores(vt, in_len)
        qkt = q.matmul(k.transpose(-2, -1))
        scaled_qkt = qkt / (self.all_head_size ** 0.5)
        softmax_qkt = scaled_qkt.softmax(-1)
        dropout_qkt = torch.nn.functional.dropout(softmax_qkt, p=p)
        v = v.view(in_len, self.num_heads, self.attention_head_size)
        output = dropout_qkt.matmul(v.transpose(-2, -1)).view(1, 1, -1)
        w = self.out_weight(output)
        return w

# Initializing the model
m = Model(1, 1)

# Inputs to the model
query = torch.randn(1, 1, 1)
key = torch.randn(1, 1, 1)
value = torch.randn(1, 1, 1)
input_tensor = torch.randn(1, 1, 64, 64)
input2 = torch.randn(1, 1, 64, 64)
p = 0.0
