
class Model(torch.nn.Module):
    def __init__(self, dropout, inv_scale_factor, n_heads, n_features_per_head):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.inv_scale_factor = inv_scale_factor
        self.n_heads = n_heads
        self.n_features_per_head = n_features_per_head
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        return torch.matmul(dropout_qk, v)
 
# Initializing the model
m = Model(dropout=0.1, inv_scale_factor=0.5, n_heads=2, n_features_per_head=4)

# Inputs to the model
q = torch.randn(1, 2, 4, 8, requires_grad=True)
k = torch.randn(1, 2, 8, 8)
v = torch.randn(1, 2, 8, 8)
