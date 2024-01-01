
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # These values must be set correctly when initializing the model
        self.input_size = 64
        self.num_heads = 8
        self.num_encoder_layers = 1
        self.num_decoder_layers = 1
        self.dim = 512
        self.dropout_p = 0.1
        self.qkv_bias = True
        self.projection_bias = True
        self.scale_factor = 1 / self.dim**0.5
 
        self.q_lin = torch.nn.Linear(self.input_size, self.dim, bias=None)
        self.k_lin = torch.nn.Linear(self.input_size, self.dim, bias=None)
        self.v_lin = torch.nn.Linear(self.input_size, self.dim, bias=none)
 
    def forward(self, query, encoder_out):
        q = self.q_lin(query).reshape(query_shape)
        k = self.k_lin(encoder_out)
        if not self.qkv_bias:
            q, k = q.div(self.scale_factor), k.div(self.scale_factor)
 
        mask = torch.ones(seq_len, seq_len).triu_(1)
        x_ = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)
 
        if self.qkv_bias:
            x = (q + self.q_bias) + (k.transpose(-2, -3) + self.k_bias)
        else:
            x = torch.matmul(q, k.transpose(-2, -1))
 
        x = x * self.scale_factor + (1 - self.scale_factor) * eye
        x = torch.nn.functional.softmax(x, -1)
        x_ = torch.matmul(x, v_)
        x_ = torch.nn.functional.dropout(x_,
                                          p=self.dropout_p)
        x_d = self.out_proj(x_)
        if self.projection_bias:
            x_d = x_d + self.out_proj_bias
        return self.ln_1(x_d)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, seq_len, self.input_size)
encoder_out = torch.randn(seq_len, batch_size, self.input_size)
