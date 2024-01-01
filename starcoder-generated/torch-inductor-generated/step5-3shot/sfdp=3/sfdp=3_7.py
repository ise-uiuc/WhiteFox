
def _init(self, embed_dim, num_heads, dropout_p=0., bias=False):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.head_dim = embed_dim // num_heads

        self.qk = torch.nn.Linear(embed_dim, embed_dim)
        self.v = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.out = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        qk = self.qk(query)
        v = self.v(value)
        qk = qk.reshape(-1, qk.shape[-2], self.num_heads, self.head_dim)
        qk = qk.permute(0, 2, 1, 3)
        v = v.reshape(-1, v.shape[-2], self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)
        k = qk.permute(0, 1, 3, 2)
        query = qk.reshape(-1, qk.shape[-2], qk.shape[-1])
        key = k.reshape(-1, k.shape[-2], k.shape[-1])
        value = v.reshape(-1, v.shape[-2], v.shape[-1])

        scale_factor = 1 / math.sqrt(query.shape[-1])
        scores = torch.matmul(query, key.transpose(-2, -1))
        scaled_scores = scores * scale_factor
        softmax_scores = scaled_scores.softmax(-1)
        self.dropout_qk = self.dropout(softmax_scores)
        output = torch.matmul(self.dropout_qk, value)
        output = output.permute(0, 2, 1, 3)
        h = output.reshape(qk.shape[0], -1, self.embed_dim)
        return self.out(h)

# Initializing the model
m = MultiHeadedAttention(num_heads=16, dropout_p=0.1, embed_dim=256)

# Inputs to the model
query = torch.randn(2, 20, 256)
key = torch.randn(2, 40, 256)
value = torch.randn(2, 40, 256)
