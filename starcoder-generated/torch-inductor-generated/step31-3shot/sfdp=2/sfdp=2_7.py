
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 2
        self.nhead = 2
        self.dropout_p = 0.0
        self.w_qkv = self.w_proj = torch.nn.Linear(self.d_model, self.d_model * 3)
        self.attention = torch.nn.MultiheadAttention(self.d_model, self.nhead, torch.nn.Dropout(self.dropout_p))
 
    def forward(self, query, key, value, inv_scale_factor):
        qkv = self.w_qkv(query).chunk(3, dim=-1)
        query = qkv[0].contiguous().view(query.size(0), -1, self.nhead, self.d_model // self.nhead)
        key = qkv[1].contiguous().view(key.size(0), -1, self.nhead, self.d_model // self.nhead)
        value = qkv[2].contiguous().view(value.size(0), -1, self.nhead, self.d_model // self.nhead)
        scaled_qk = self.attention(query, key, value, need_weights=False, attn_mask=None)[0]
        softmax_qk = scaled_qk * (self.d_model // self.nhead) ** -0.5
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.transpose(1, 2).contiguous().view(query.size(0), -1, self.nhead * self.d_model // self.nhead)
        output = self.w_proj(output)
        return output, dropout_qk
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 2)
key = torch.randn(1, 5, 2)
value = torch.randn(1, 5, 2)
inv_scale_factor = 1. / math.sqrt(2)
__output__, dout = m(query, key, value, inv_scale_factor)
```

# References
1. https://discuss.pytorch.org/t/how-to-get-the-model-summary/455
2. https://stackoverflow.com/questions/50126264/how-does-one-generate-model-summary-in-pytorch
3. https://towardsdatascience.com/model-summary-in-pytorch-47d7a8459168
4. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py
5. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/models/resnet.py
6. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L300
7. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L397
8. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L608
9. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L661
10. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L642
11. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L610-L611
12. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L716
13. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L754
14. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L715
15. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L613-L614
16. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L767-L770
17. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/torchsummary.py#L44
18. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/models/resnet.py#L25
19. https://github.com/sksq96/pytorch-summary/blob/669a381341bf6f1e13ad16b1c3e5c66429d26222/torchsummary/models/resnet.py#L354