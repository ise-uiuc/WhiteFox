
class Model(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Assume that the dimensions of the query, key, and value are equal to each other
        n_heads = 1
        dim = 28
        n = 12
        self.in_proj_weight_qkv = torch.nn.Parameter(torch.randn(n_heads, dim, n))
        self.pos_proj_weight_qkv = torch.nn.Parameter(torch.randn(n_heads, dim, n))
        self.in_proj_bias_qkv = torch.nn.Parameter(torch.randn(n_heads, dim))
        self.pos_proj_bias_qkv = torch.nn.Parameter(torch.randn(n_heads, dim))
        # Assume that dropout rate is 0.5
        dropout_p = 0.5
        scale_factor_init = math.sqrt(n)
        self.pos_proj_weight_dropout = torch.nn.Parameter(torch.from_numpy(
            np.full((dim, dim), 1 / scale_factor_init)))
        self.pos_proj_bias_dropout = torch.nn.Parameter(torch.full((dim, ), 1 / scale_factor_init))
 
    def _scaled_matmul(self, x1, weight, scale):
        return x1.matmul(weight).mul(scale)
 
    def forward(self, query, key, value):
        head_dim = query.shape[-1]
        # For query, key, and value, compute the dot product and scale by an inverse scale factor
        in_proj_weight_q, in_proj_weight_k, in_proj_weight_v = self.in_proj_weight_qkv.chunk(3)
        q, k, v = query.matmul(in_proj_weight_q).div(head_dim), key.matmul(
            in_proj_weight_k).div(head_dim), value.matmul(in_proj_weight_v).div(head_dim)
        # For query, key, and value, apply dropout
        dropout_p = 0.5
        pos_proj_weight_dropout = self.pos_proj_weight_dropout.div(head_dim)
        pos_proj_bias_dropout = self.pos_proj_bias_dropout.div(head_dim)
        q, k, v = self._scaled_matmul(q, pos_proj_weight_dropout, pos_proj_bias_dropout),\
                  self._scaled_matmul(k, pos_proj_weight_dropout, 
                                     pos_proj_bias_dropout), self._scaled_matmul(
                v, pos_proj_weight_dropout, pos_proj_bias_dropout)
        # Compute the dot product of the query and the key, also apply dropout
        qk = q.matmul(k.transpose(-2, -1))
        dropout_p = 0.5
        pos_proj_weight_qkv = self.pos_proj_weight_qkv.div(head_dim)
        pos_proj_bias_qkv = self.pos_proj_bias_qkv.div(head_dim)
        qk = self._scaled_matmul(qk, pos_proj_weight_qkv, pos_proj_bias_qkv)
        # Compute softmax
        softmax_qk = qk.softmax(dim=-1)
        # Apply dropout to softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        # Compute the dot product of the dropout output and the value
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
num_qkv_attention_heads = 2
dim = 3
n = 5
dropout_p = 0.5
m = Model(hparams={"num_attention_heads": num_qkv_attention_heads, "dim": dim*num_qkv_attention_heads, "dropout_p": dropout_p, "n": n})

# Inputs to the model
batch_size = 3
query = torch.randn(batch_size, num_qkv_attention_heads, dim*num_qkv_attention_heads, n)
key = torch.randn(batch_size, num_qkv_attention_heads, dim*num_qkv_attention_heads, n)
value = torch.randn(batch_size, num_qkv_attention_heads, dim*num_qkv_attention_heads, n)
