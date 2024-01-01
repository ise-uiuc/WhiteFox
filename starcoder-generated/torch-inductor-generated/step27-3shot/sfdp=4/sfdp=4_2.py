
class Model(torch.nn.Module):
    def __init__(self, n_head=1):
        super().__init__()
        self.n_head = n_head
        self.linear_q = torch.nn.Linear(64, 64 * self.n_head)
        self.linear_k = torch.nn.Linear(64, 64 * self.n_head)
        self.linear_v = torch.nn.Linear(64, 64 * self.n_head)
        self.linear_out = torch.nn.Linear(64 * self.n_head, 64)
 
    def forward(self, input_tensor, attn_mask=None):
        query = self.linear_q(input_tensor)
        key = self.linear_k(input_tensor)
        value = self.linear_v(input_tensor)
        query = query.view(query.shape[0], query.shape[1] * self.n_head, query.shape[2])
        key = key.view(key.shape[0], key.shape[1] * self.n_head, key.shape[2])
        value = value.view(value.shape[0], value.shape[1] * self.n_head, value.shape[2])
        qk = query @ key.transpose(-2, -1) / math.sqrt(64)
        if attn_mask is not None:
            qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_out = attn_weight @ value
        attn_out = attn_out.view(attn_out.shape[0], attn_out.shape[1], attn_out.shape[2] // self.n_head)
        attn_out = self.linear_out(attn_out)
        return attn_out

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 64, 128)
attn_mask = torch.zeros(1, 9, 9)  # 128 = 8 * 8, where 8 = self.n_head * 2^(n_layer - 1)
attn_mask[0, 6:9, 0:3] = float("-inf")
attn_mask[0, 0:3, 6:9] = float("-inf")
