
class FeedForward(torch.nn.Module):
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(128)
        self.ln2 = torch.nn.LayerNorm(128)
        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(256, 128)
        self.dropout_p = 0.1
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.ff1 = FeedForward()
 
    def forward(self, query, key, value, inv_scale_factor):
        query = self.ln1(query)
        key = self.ln2(key)
        v1 = self.dense1(query)
        v2 = self.dense2(v1)
        v3 = self.dropout(v2)
        output = self.ff1(query, key, value, inv_scale_factor, self.dropout_p)
        return output

# Initializing the model
inv_scale_factor = 8.0
m = Model()

# Inputs to the model
query = torch.randn(1, 256, 128)
key = torch.randn(1, 256, 128)
value = torch.randn(1, 256, 128)
