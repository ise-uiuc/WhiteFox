
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(
            dim_q = 8, dim_k = 8, dim_v = 8, num_heads = 2, dropout_p = 0.1, scale_factor = 1 / (8 ** 0.5)
        )
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, q, k, v):
        q = q.chunk(self.num_heads, dim=1)
        k = k.chunk(self.num_heads, dim=1)
        v = v.chunk(self.num_heads, dim=1)
        q = [q_i.transpose(-2, -1) for q_i in q]
        k = [k_i.transpose(-2, -1) for k_i in k]
        v = [v_i.transpose(-2, -1) for v_i in v]
        q_ = [self.w_q([q_i]) for q_i in q]
        k_ = [self.w_k([k_i]) for k_i in k]
        v_ = [self.w_v([v_i]) for v_i in v]
        attn = [torch.matmul(q_i, k_i) for (q_i, k_i) in zip(q_, k_)]
        scaled_attn = [self.dropout(attn_i) * self.scale_factor for attn_i in attn]
        softmax_attn = [torch.nn.functional.softmax(attn_i, dim=-1) for attn_i in scaled_attn]
        output = [softmax_attn_i[-1].matmul(value_i) for (softmax_attn_i, value_i) in zip(softmax_attn, v_)]
        output_ = [
            output_i.transpose(1, 2) for output_i in output
        ]
        output = [
            torch.cat([output_i_h[i].unsqueeze(1) for output_i_h in output_i], dim=1)
            for (i, output_i) in enumerate(output_)
        ]
        output = torch.cat(output, dim=2)
        return output
 
class W(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out)
 
    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.hardtanh(x)
        x = x.transpose(-2, -1)
        x = x.softmax(dim=-1)
        return x
 
class Attention(torch.nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads=1, dropout_p=0., scale_factor=1.0):
        super().__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.w_q = W(dim_q, dim_q)
        self.w_k = W(dim_k, dim_k)
        self.w_v = W(dim_v, dim_v)
        self.dropout_p = dropout_p
 
    def forward(self, q, k, v=None, input_mask=None):
        if v is None:
            v = k
        if input_mask is not None:
            input_mask = (1.0 - input_mask) * -10000
        output = self.forward_impl(q, k, v, input_mask)
        return output
 
    def forward_impl(self, q, k, v, input_mask):
        q_chunk = q.chunk(self.num_heads, dim=-2)
        k_chunk = k.chunk(self.num_heads, dim=-2)
        v_chunk = v.chunk(self.num_heads, dim=-2)
        q_chunk = [q_i.transpose(-2, -1) for q_i in q_chunk]
        k_chunk = [k_i.transpose(-2, -1) for k_i in k_chunk]
        v_chunk = [v_i.transpose(-2, -1) for v_i in v_chunk]
        q_ = [self.w_q([q_i]) for q_i in q_chunk]
        k_ = [self.w_k([k_i]) for k_i in k_chunk]
        v_ = [self.w_v([v_i]) for v_i in v_chunk]
        attn = [torch.matmul(q_i, k_i) for (q_i, k_i) in zip(q_, k_)]
        scaled_attn = [attn_i * self.scale_factor + input_mask for attn_i in attn]
        softmax_attn = [torch.nn.functional.softmax(scaled_attn_i, dim=-1) for scaled_attn_i in scaled_attn]
        softmax_attn = [
            self.dropout(softmax_attn_i) if self.training else softmax_attn_i
            for softmax_attn_i in softmax_attn
        ]
        output = [softmax_attn_i[-1].matmul(value_i) for (softmax_attn_i, value_i) in zip(softmax_attn, v_)]
        output_ = [
            output_i.transpose(1, 2) for output_i in output
        ]
        output = [
            torch.cat([output_i_h[i].unsqueeze(1) for output_i_h in output_i], dim=1)
            for (i, output_i) in enumerate([output_])
        ]
        output = torch.cat(output, dim=2)
        return output
     
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(
            dim_q=8, dim_k=8, dim_v=8, num_heads=2, dropout_p=0.1, scale_factor=1 / (8 ** 0.5), pad_token_id=1,
        )
        self.attn = Attention(dim_q=self.dim_q, dim_k=self.dim_k, dim_v=self.dim_v)
        self.dense = torch.nn.Linear(self.dim_v, 2)
 
    def forward(self, x1, x2):
        x2 = self.pad_token_id * torch.ones(1, 2, 2).to(x1.device) - x2.float()
        x1 = self.attn(x1, x2)
        x1 = self.dense(x1)
        return x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 1)
