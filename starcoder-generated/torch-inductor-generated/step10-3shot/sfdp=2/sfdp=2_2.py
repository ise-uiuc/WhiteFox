
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads, head_dim, dropout_p):
        super().__init__()
        self.w_q = torch.nn.Linear(dim, num_heads * head_dim)
        self.w_k = torch.nn.Linear(dim, num_heads * head_dim)
        self.w_v = torch.nn.Linear(dim, num_heads * head_dim)
        self.w_o = torch.nn.Linear(num_heads * head_dim, dim)
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = self.head_dim ** -0.5
 
    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.reshape([q.shape[0], q.shape[1], self.num_heads, self.head_dim])
        k = k.reshape([k.shape[0], k.shape[1], self.num_heads, self.head_dim])
        v = v.reshape([v.shape[0], v.shape[1], self.num_heads, self.head_dim])
        q *= self.scaling
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(-2, -1)
        attn_score = q @ k
        attn_p = torch.nn.functional.dropout(
            attn_score.softmax(dim=-1), p=self.dropout_p
        )
        attn_output = attn_p @ v
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape([attn_output.shape[0], -1, self.num_heads * self.head_dim])
        return self.w_o(attn_output)

# Initializing the model
m = Model(dim=64, num_heads=2, head_dim=16, dropout_p=0.1)

# Inputs to the model
x = torch.randn(64, 16, 64)
