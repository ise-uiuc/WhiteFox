
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, dropout_p, scale_factor):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.scale_factor = scale_factor
        self.qk_projection = torch.nn.Linear(query_size, key_size)
        self.v_projection = torch.nn.Linear(value_size, key_size)
 
    def forward(self, query, key, value):
        qk = self.qk_projection(query).matmul(self.v_projection(key).transpose(-2, -1))
        scale_qk = qk / self.scale_factor
        soft_qk = self.softmax(scale_qk)
        dropout_qk = self.dropout(soft_qk)
        return dropout_qk.matmul(value)

# Initializing the model
dropout_p = 0.8
scale_factor = 1.2
m = Model(query_size=128, key_size=256, value_size=512, dropout_p=dropout_p, scale_factor=scale_factor)

# Inputs to the model
query = torch.randn(32, 128)
key = torch.randn(32, 256)
value = torch.randn(32, 512)
