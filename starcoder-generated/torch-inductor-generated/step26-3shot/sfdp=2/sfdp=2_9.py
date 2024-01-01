
class Model(torch.nn.Module):
    def __init__(self, n_queries=8, n_keys=8, n_values=8, n_head=8, dropout_p=0.2):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(n_head, n_queries, 101))
        self.key = torch.nn.Parameter(torch.randn(n_head, n_keys, 101))
        self.value = torch.nn.Parameter(torch.randn(n_head, n_values, 101))
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x2):
        q = self.query.unsqueeze(0).unsqueeze(0).contiguous()
        k = self.key.unsqueeze(0).contiguous()
        v = self.value.unsqueeze(0).contiguous()
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1 / np.sqrt(np.prod(qk.shape[-2:]))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output.squeeze(0)

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
