
class Model(torch.nn.Module):
    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        self.q = torch.nn.Linear(feature_dim, feature_dim)
        self.k = torch.nn.Linear(feature_dim, feature_dim)
        self.v = torch.nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q, k, v = [x.view(x.size(0), x.size(1), num_heads, -1) for x in [q, k, v]]
        q, k, v = [x.transpose(1, 2) for x in [q, k, v]]
        
        qk = torch.matmul(q, k)
        
        scale_factor = (k.size(-1) / qk.size(-1)) ** -0.5

        softmax_qk = F.softmax(qk * scale_factor, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout_rate)

        output = torch.matmul(dropout_qk, v)

        # convert to [batch, seq_len, dmodel]
        output = output.transpose(1, 2)
        output = output.contiguous()
        output = output.view(output.size(0), output.size(1), -1)
        return output

# Initializing the model
m = Model(feature_dim=512, num_heads=8)

# Inputs to the model
x = torch.randn(16, 64, 512)
