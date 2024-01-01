
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_func = torch.nn.MultiheadAttention(d_model=512, nhead=8)
        self.dropout_p = 0.2
    def forward(self, x):
        v1 = x.permute(1, 0, 2)
        v2, v3, v4 = self.attn_func(v1, v1, v1, attn_mask=None, key_padding_mask=None, need_weights=True)
        del v1, v4
        v5 = v2.permute(1, 0, 2)
        v6 = v5.div(512**0.5)
        del v5
        v7 = torch.nn.functional.softmax(v6, dim=-1)
        del v6
        v8 = torch.nn.functional.dropout(v7, p=self.dropout_p)
        del v7
        v9 = v8.matmul(v3.permute(1, 0, 2).float())
        del v3
        v10 = v9.float()
        del v8, v9
        v11 = v10[0].permute(1, 0, 2)
        return v11
torch.manual_seed(0)
m = Model()

x = torch.randn(1, 1024, 512)
