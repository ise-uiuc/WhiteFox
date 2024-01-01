
class Model(torch.nn.Module):
    def __init__(self, num_queries, num_keys, num_values, num_heads=8, scale_factor=1.0, dropout_p=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale_factor = scale_factor / num_heads ** 0.5
        self.dropout_p = dropout_p
        self.matmul1 = torch.nn.Linear(in_features=num_queries, out_features=num_heads * num_keys, bias=False)
        self.matmul2 = torch.nn.Linear(in_features=num_keys, out_features=num_heads * num_values, bias=False)
 
    def forward(self, query, key, value):
        matmul1 = self.matmul1(query)
        matmul2 = self.matmul2(key)
        qh = matmul1.view(matmul1.shape[:-1] + (self.num_heads, self.num_keys))
        kh = matmul2.view(matmul2.shape[:-1] + (self.num_heads, self.num_values))
        qk = torch.matmul(qh, kh.permute(0, 1, 3, 2))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model(num_queries=32, num_keys=64, num_values=64)
 
# Inputs to the model
q = torch.randn(2, 32, 512)
k = torch.randn(2, 64, 512)
v = torch.randn(2, 64, 512)
