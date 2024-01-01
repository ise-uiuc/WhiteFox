
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, channel_dim):
        super().__init__()
        self.scale_factor = torch.zeros(1)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor.sqrt())
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
model = Model(query_dim=1024, key_dim=512, channel_dim=1024)

# Inputs to the model
query = torch.randn(1, 64, 1024)
key = torch.randn(1, 64, 512)
value = torch.randn(1, 64, 1024)
