
class Model(torch.nn.Module):
    def __init__(self, num_features, dropout_p, num_heads, hidden_dims):
        super().__init__()
        self.num_features = num_features
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
 
        self.qk_linear = torch.nn.Linear(self.num_features, self.num_features)
        self.v_linear = torch.nn.Linear(self.num_features, self.num_features)
 
    def forward(self, query, key, value, inv_scale_factor=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        if inv_scale_factor is not None:
            qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(num_features=128, dropout_p=0.1, num_heads=4, hidden_dims=256)

# Inputs to the model
x1 = torch.randn(4, 64, 128)
x2 = torch.randn(4, 256, 128)
x3 = torch.randn(4, 64, 128)
