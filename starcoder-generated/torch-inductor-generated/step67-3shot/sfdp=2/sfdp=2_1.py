
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = torch.nn.Sequential(
            torch.nn.Linear(65536, 2048),
            torch.nn.Linear(2048, 65536),
            torch.nn.LayerNorm(65536),
            torch.nn.Gelu(),
            torch.nn.Dropout(p=0.1),
        )
    
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Queries, keys, values, and scale factors to the model
query = torch.randn(1, 55, 65536)
key = torch.randn(1, 55, 65536)
value = torch.randn(1, 55, 65536)
inv_scale_factor = 0.1
