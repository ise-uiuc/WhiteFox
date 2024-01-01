 (cont'd)
class Model(torch.nn.Module):
    def _init__(self, num_heads, dim_per_head):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.query_projection = torch.nn.Linear(dim, self.num_heads * self.dim_per_head)
        self.key_projection = torch.nn.Linear(dim, self.num_heads * self.dim_per_head)
        self.value_projection = torch.nn.Linear(dim, self.num_heads * self.dim_per_head)
 
    def forward(self, query, key, value, dropout_p, mask):
        