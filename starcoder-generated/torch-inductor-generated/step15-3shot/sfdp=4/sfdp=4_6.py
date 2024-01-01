
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x, attn_mask):
        v1 = self.conv(x)
        qk1 = qk[:, 0*dim_per_head : 1*dim_per_head] @ k[:, 0*dim_per_head : 1*dim_per_head] # q[:, 0*dim_per_head : 1*dim_per_head] or q[:, 0*dim_per_head : 1*dim_per_head, :, :]
        qk2 = qk[:, 1*dim_per_head : 2*dim_per_head] @ k[:, 1*dim_per_head : 2*dim_per_head] # q[:, 1*dim_per_head : 2*dim_per_head] or q[:, 1*dim_per_head : 2*dim_per_head, :, :]
        qk3 = qk[:, 2*dim_per_head : 3*dim_per_head] @ k[:, 2*dim_per_head : 3*dim_per_head] # q[:, 2*dim_per_head : 3*dim_per_head] or q[:, 2*dim_per_head : 3*dim_per_head, :, :]
    a2 = v + v
        qk = torch.cat([qk1, qk2, qk3], dim=1)
        qk = torch.cat([qk1, qk2, qk3], dim=1)
        qk = torch.nn.functional.softplus(qk)
        qk = torch.nn.functional.softmax(qk, dim=-1)
