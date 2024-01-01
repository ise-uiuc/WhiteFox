
class Model(torch.nn.Module):
    def __init__(self, feature_size, num_heads, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(feature_size, feature_size)
        self.key = torch.nn.Linear(feature_size, feature_size)
        self.value = torch.nn.Linear(feature_size, feature_size)
        self.scaled_factor = math.sqrt(feature_size)

    def forward(self, query, key, value, scale_factor):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(v)
 
# Initializing the model
m = Model(feature_size=256, num_heads=4, dropout_p=0.1)

# Inputs to the model
query = torch.randn(8, 32, 256)
key = torch.randn(8, 64, 256)
value = torch.randn(8, 64, 256)
scale_factor = math.sqrt(256)
