
class Model(torch.nn.Module):
    def __init__(self, num_head, hidden_dim):
        super().__init__()
        self.num_head = num_head
        self.head_dim = d_k = hidden_dim // num_head
        self.scaling = d_k**-0.5
        self.q_net = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_net = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_net = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attn_net = torch.nn.Linear(hidden_dim, 1)
        self.proj_net = torch.nn.Linear(hidden_dim, hidden_dim)
 
    def forward(self, x, x_mask):
        q = self.q_net(x)
        k = self.k_net(x)
        v = self.v_net(x)
        batch_size, n_seq, d_model = x.size()
        # reshape qkv for multi-head attention
        q = q.view(batch_size, n_seq, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_seq, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_seq, self.num_head, self.head_dim).transpose(1, 2)
        attn_mask = -torch.ones(n_seq, n_seq).to(x.device)
        if x_mask is not None:
            attn_mask = attn_mask.masked_fill(x_mask.to(torch.bool), float('-inf'))
        # scale dot product attention
        attn_weight = torch.softmax(self.scaling
                        * self.attn_net(torch.nn.functional.leaky_relu(q @ k.transpose(-2, -1), 0.1)).transpose(1, 2), -1)
        output = attn_weight @ v   
        output = output.transpose(1, 2).reshape(batch_size, n_seq, d_model)
        output = torch.tanh(self.proj_net(output))
        return output

# Initializing the model
m = Model(num_head=8, hidden_dim=64)

# Inputs to the model
x = torch.randn(3, 10, 64)
x_mask = torch.tensor([[1, 1, -1, -1, -1, -1, -1, -1, -1, -1]])
