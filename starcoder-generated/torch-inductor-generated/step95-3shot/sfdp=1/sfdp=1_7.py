
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x1, x2):
        q = self.linear(x1)
        k = self.linear(x1)
        v = self.linear(x2)
  
        qk = torch.matmul(q, k.transpose(-2, -1))
        rands = torch.randint(0, 1, shape=qk.shape, device=x1.device)
        mask = torch.logical_not(torch.equal(rands, 0))
        inv_scale_factor = torch.div(1.0, mask.float().sum(axis=-1).unsqueeze(1))
        scaled_qk = torch.div(qk, inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        dropout_qk = dropout_qk * mask
        output = torch.matmul(dropout_qk, v)
        return output


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 10, 16)
x2 = torch.randn(1, 4, 30, 16)
