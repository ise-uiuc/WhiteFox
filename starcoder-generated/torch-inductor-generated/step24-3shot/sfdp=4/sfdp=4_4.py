
import math

class Model(torch.nn.Module):
    def __init__(self, q_in_features, k_in_features, num_heads):
        super().__init__()
        self.head_dim = q_in_features // num_heads
        self.q_linear = torch.nn.Linear(q_in_features, self.head_dim*num_heads)
        self.k_linear = torch.nn.Linear(k_in_features, self.head_dim*num_heads)
        self.v_linear = torch.nn.Linear(k_in_features, self.head_dim*num_heads)
        self.out_linear = torch.nn.Linear(q_in_features, q_in_features)
 
    def forward(self, q, k, v, attn_mask):
        q = self.q_linear(q).view(q.shape[0], q.shape[1], q.shape[2], self.num_heads, -1)
        k = self.k_linear(k).view(k.shape[0], k.shape[1], k.shape[2], self.num_heads, -1)
        v = self.v_linear(v).view(v.shape[0], v.shape[1], v.shape[2], self.num_heads, -1)
        q = q.permute(0, 3, 1, 2, 4)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scaled_dot_product = scaled_dot_product + attn_mask
        attn_weight = torch.softmax(scaled_dot_product, dim=-1)
        output = torch.matmul(attn_weight, v)
        output = output.permute(0, 2, 1, 3, 4).contiguous().view(
            output.shape[0], -1, q.shape[3], q.shape[4]
        )
        output = self.out_linear(output)
        return output

# Initializing the model
m = Model(q_in_features=2, k_in_features=1, num_heads=2)

# Inputs to the model
q = torch.randn(1, 2, 4, 1)
k = torch.randn(1, 1, 6, 1)
v = torch.randn(1, 1, 6, 1)
attn_mask = torch.tensor([[[[1, 0]]]], dtype=torch.float)
