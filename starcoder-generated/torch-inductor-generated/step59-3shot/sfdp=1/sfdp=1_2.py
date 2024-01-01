
class Model(torch.nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.query = torch.nn.Linear(128, n_head * 32)
        self.key = torch.nn.Linear(32, n_head * 32)
        self.value = torch.nn.Linear(32, n_head * 32)
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        q /= x1.shape[-1]
        q = q.view(-1, q.shape[-2], q.shape[-1])
        k = k.view(-1, k.shape[-2], k.shape[-1])
        v = v.view(-1, v.shape[-2], v.shape[-1])
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk / k.shape[-1]
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model(n_head=2)

# Inputs to the model
x1 = torch.randn(2, 128)
x2 = torch.randn(2, 32)
