
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = torch.nn.Linear(8, 8, bias=False)
        self.w_k = torch.nn.Linear(8, 8, bias=False)
        self.w_v = torch.nn.Linear(8, 8, bias=False)
 
    def forward(self, q, k, v):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        k = k.transpose(-2, -1)
        logits = torch.matmul(q, k)
        scale_factor = 1.0 / np.sqrt(logits.size(-1))
        scaled_logits = logits * scale_factor
        softmax_logits = torch.nn.functional.softmax(scaled_logits, dim=-1)
        dropout_logits = torch.nn.functional.dropout(softmax_logits, p=0.5)
        return torch.matmul(dropout_logits, v)

# Input for two query tensor, key tensor, and value tensor
q = torch.randn(1, 4, 8)
k = torch.randn(1, 3, 8)
v = torch.randn(1, 3, 8)

# Initializing the model
m = Model()

# Inputs to the model
