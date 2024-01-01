
class Model(torch.nn.Module):
    def __init__(self, num_heads, head_size, input_length, device):
        super().__init__()
        self.w = torch.nn.Linear(head_size * 2, num_heads * head_size).to(device)
        self.b = torch.nn.Parameter(torch.zeros(num_heads * head_size, requires_grad=True)).to(device)
 
    def forward(self, q, k, v, dropout_p):
        scale_factor = torch.tensor(math.sqrt(math.sqrt(q.shape[-1])))
        inv_scale_factor = 1.0 / scale_factor
        head_dim = q.shape[-1]
        query = q.contiguous().view(-1, q.shape[-2], q.shape[-1])
        key = k.contiguous().view(-1, k.shape[-2], k.shape[-1]).transpose(-2, -1)
        value = v.contiguous().view(-1, v.shape[-2], v.shape[-1])
        
        qk = torch.matmul(query, key)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        
        output = dropout_qk.matmul(value)
        
        output = output.view(-1, head_dim, output.shape[-2], output.shape[-1])
        output = output.transpose(-3, -2)

        output = output.contiguous().view(q.shape[0], q.shape[1], q.shape[2], q.shape[3])
        output = self.w(output.reshape(output.shape[0], output.shape[1], output.shape[2] * output.shape[3])).reshape(q.shape[0], head_dim, q.shape[2], q.shape[3])
        output = output + self.b.view(1, -1, 1, 1)
        output = torch.relu(output)
        output = output.reshape(q.shape[0], q.shape[1] * q.shape[2], q.shape[3])
        
        return output
        
        
 
# Initializing the model
m = Model(2, 2, 100, 'cpu')

# Inputs to the model
q = torch.randn(8, 2, 2520)
k = torch.zeros(8, 2, 2520)
v = torch.ones(8, 2, 2520)
dropout_p = 0.5
