
class Model(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout_p=0.0, inv_scale_factor=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.qk_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query, key, value, attn_mask):
        qk = torch.matmul(query, self.qk_projection(key).transpose(-2, -1))
        qk *= self.inv_scale_factor
        output = torch.matmul(self.softmax(qk), self.dropout(value))
        return output.transpose(0, 1)
 
# Initializing the model
query = torch.randn(4, 2, 8)
key = torch.randn(4, 2, 16)
value = torch.randn(4, 2, 16)
attn_mask = np.ones((4, 2, 2))
attn_mask = 1 - attn_mask
