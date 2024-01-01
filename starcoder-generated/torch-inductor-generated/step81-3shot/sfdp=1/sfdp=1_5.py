
class Model(torch.nn.Module):
    def __init__(self, d_model, head_num, dropout_p=0.7):
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.dropout_p = dropout_p
        assert (d_model // head_num) * head_num == d_model
        self.qs = torch.nn.Linear(d_model, d_model, bias=False)
        self.ks = torch.nn.Linear(d_model, d_model, bias=False)
        self.vs = torch.nn.Linear(d_model, d_model, bias=False)
        self.head_split = lambda x: torch.split(x, self.d_model // self.head_num, dim=-1)
            
    def forward(self, query, key, value, dropout_p=0.7, attention_mask=None):
        d_k = query.shape[-1] // self.head_num
        q_heads = self.head_split(self.qs(query))
        k_heads = self.head_split(self.ks(key))
        v_heads = self.head_split(self.vs(value))
        qs = [torch.einsum('...id,...jd->...ij', q_head, k_head) for q_head, k_head in zip(q_heads, k_heads)]
        scaled_qs = [q / np.sqrt(d_k) for q, d_k in zip(qs, [d_k for _ in range(self.head_num)])]
        softmax_qs = [torch.nn.functional.softmax(q, dim=-1) for q in scaled_qs]
        dropout_qs = [torch.nn.functional.dropout(sq, p=dropout_p) for sq in softmax_qs]
        attention = [q * torch.einsum('...ij,...jd->...id', dropout_q, v_head) for q, dropout_q, v_head in zip(dropout_qs, softmax_qs, v_heads)]
        output = torch.stack(attention, dim=0).sum(dim=0)
        return output

# Initialize a model
m = Model(512, 8)

# Input to the model
query = torch.randn(1, 10, 512)
key = torch.randn(1, 10, 512)
value = torch.randn(1, 10, 512)

# Optional parameter
dropout_p = 0.7 

# Optional mask input
attention_mask = [[[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]]
attention_mask = torch.tensor(attention_mask)
