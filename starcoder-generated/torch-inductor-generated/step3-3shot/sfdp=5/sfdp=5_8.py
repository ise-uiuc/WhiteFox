
class Model(torch.nn.Module):
    __output_padding__ = [0, 1, 1, 0]
    
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.0
        self.query = [[[0.5, 0.7]]]
        self.key = [[[0.1, 0.2]]]
        self.value = [[[0.3, 0.4]]]
        self.attn_mask = torch.nn.Parameter(torch.tensor([[[0.0, float("-inf")]]]), requires_grad=True)
        self.proj = torch.nn.Linear(1, 1)
        
    def forward(self, x1):
        v7 = torch.nn.functional.dropout(self.proj(self.query), self.dropout_p, True)
        v8 = v7 @ torch.tensor([[0.5]]) + self.attn_mask
        v9 = v8.permute(0, 2, 3, 1)
        v10 = torch.nn.functional.softmax(v9, dim=1)
        v11 = torch.nn.functional.dropout(v10, self.dropout_p, True)
        v12 = v11 * self.value
        v13 = v12.sum(dim=1)
        t14 = v13 + v13 + v13
        v15 = t14 - t14 * t14
        