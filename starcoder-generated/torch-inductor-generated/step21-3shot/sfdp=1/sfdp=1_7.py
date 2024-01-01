
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.10000000000000001)
 
    def forward(self, query, key, value, attn_mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(2048.0)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(128, 4, 1024)
key = torch.randn(128, 1024, 512)
value = torch.randn(128, 1024, 512)
attn_mask = torch.randn(128, 5, 32)
