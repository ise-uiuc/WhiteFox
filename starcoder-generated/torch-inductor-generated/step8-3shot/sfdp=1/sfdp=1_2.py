
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def init_weights(self):
        for p in self.parameters():
            p.data.normal_(std=0.001)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p=0.5, training=False, inplace=False):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        return torch.matmul(dropout_qk, value)

# Initializing the model
m = Model()
m.apply(Model.init_weights)

# Inputs to the model
query = torch.randn(16, 256, 32)
key = torch.randn(16, 32, 256)
value = torch.randn(16, 32, 256)
inv_scale_factor = torch.randn(1)
