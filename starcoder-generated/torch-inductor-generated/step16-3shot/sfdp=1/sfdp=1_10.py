
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
 
    def forward(self, hidden_states):
        q = hidden_states[:, :self.hidden_size]
        k = hidden_states[:, self.hidden_size:]
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = k.size(-2)**.5
        qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(hidden_size=64)

# Inputs to the model
x = torch.randn(1, 128, 64)
