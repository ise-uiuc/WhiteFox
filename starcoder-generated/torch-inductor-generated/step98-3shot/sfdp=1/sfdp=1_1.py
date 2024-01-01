
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key_value = torch.nn.Linear(3, 16)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(0.3)
 
    def forward(self, input):
        qkv = self.query_key_value(input)
        batch, head, n, _ = qkv.shape
        qkv = qkv.view(batch, head, n, 3, 4)
        q, k, v = qkv.unbind(dim=-2)
        q, k, v = map(LambdaArgmax(dim=-1), (q, k, v))
        q, k = map(lambda x: x.permute(0, 1, 3, 2), (q, k))
        scaled_qk = torch.matmul(q, k)
        inv_scale_factor = float(n ** -0.25)
        scaled_qk = scaled_qk * inv_scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        dropout_qk = dropout_qk.permute(0, 1, 3, 2)
        return dropout_qk.matmul(v)


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 15, 3)
