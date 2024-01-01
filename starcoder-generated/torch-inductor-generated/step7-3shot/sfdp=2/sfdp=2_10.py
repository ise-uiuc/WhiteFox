
class Model(torch.nn.Module):
    __inv_scale_factor__ = 1/math.sqrt(64)
    drop_path_func = DropPathFunc(drop_path_prob=0.7)
    def __init__(self, num_queries=100, seq_length=256, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(seq_length, seq_length, bias=False)
        self.k = nn.Linear(seq_length, seq_length, bias=False)
        self.v = nn.Linear(seq_length, seq_length, bias=False)
        self.o = nn.Linear(seq_length, seq_length, bias=True)

    def forward(self, queries, keys, values):
        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)
        q /= math.sqrt(math.sqrt(self.num_heads))
        x = self.drop_path_func(self.o(self.attention(q, k, v)))
        return x

    def attention(self, query, key, value):
        