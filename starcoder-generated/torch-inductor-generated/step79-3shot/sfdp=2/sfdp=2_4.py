
class Model(torch.nn.Module):
    def __init__(self, nhead=8):
        super().__init__()
        self.nhead = nhead
        self.head_dim = 8
        self.all_head_dim = self.head_dim * self.nhead

    def forward(self, query, key, value, dropout_p=0.5, inv_scale_factor=1.0 / math.sqrt(128)):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 5, 3, 8)
key = torch.randn(4, 4, 8, 8)
value = torch.randn(4, 5, 8, 64)
dropout_p=0.5
inv_scale_factor=1.0 / math.sqrt(128)
