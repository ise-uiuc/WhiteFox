

class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout_p):
        super().__init__()
        self.w_qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.dropout_p = dropout_p
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def forward(self, query, key, value):
        qkv = self.w_qkv(query)
        qkv = qkv.reshape(qkv.shape[:-1] + (self.num_heads, 3 * self.d_k)).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        inv_scale_factor = float(self.d_model) ** -0.5
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = scaled_qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, v).transpose(1, 2).reshape(output.shape[:-1] + (d_model,))
        return output

# Initializing the model
m = Model(d_model=64, num_heads=2, dropout_p=0.25)

# Inputs to the model
