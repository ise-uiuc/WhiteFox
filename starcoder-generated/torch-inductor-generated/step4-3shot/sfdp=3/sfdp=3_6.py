
class DropoutAttention2D(torch.nn.Module):
    def __init__(self, dropout_p = 0.2, dropout_dim = -1, scale_factor = 1.0 / np.sqrt(64),):
        super().__init__()
        self.dropout_p = dropout_p
        self.dropout_dim = dropout_dim
        self.scale_factor = scale_factor
 
    def forward(self, q, k, v):
        query = q
        key = k
        value = v
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=self.dropout_dim)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = DropoutAttention2D()

# Inputs to the model
q = torch.randn(128, 50, 14, 14)
k = torch.randn(128, 50, 14, 14)
v = torch.randn(128, 50, 14, 14)
