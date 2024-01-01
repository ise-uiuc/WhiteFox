
class Model(torch.nn.Module):
    def __init__(self, n_feature_keys, n_hidden_keys):
        super().__init__()
        self.n_feature_keys = n_feature_keys
        self.n_hidden_keys = n_hidden_keys
        self.n_head = 4
        
        self.w_keys = torch.nn.Parameter(torch.ones([self.n_head, self.n_feature_keys]))
        self.w_values = torch.nn.Parameter(torch.ones([self.n_head, self.n_hidden_keys]))
        self.w_inv_scale = torch.nn.Parameter(torch.ones([1]))
    
    def forward(self, data, keys, queries, mask):
        k = keys.shape[-1]
        v = data.shape[-1]
        n_batch = data.shape[0]
        n_query = queries.shape[1]
        n_head = self.n_head  # Number of heads
        head_dim = v // n_head  # Values per head
        q = torch.reshaping(queries, [n_batch * n_query, 1, -1]).transpose(-1, -2)
        k = torch.reshaping(keys, [n_batch, n_query, -1])
        v = torch.reshaping(data, [n_batch, n_query, v])
        
        inv_scale = torch.sqrt(self.w_inv_scale)
        logits = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        logits.masked_fill_(mask, maskvalue)
        attn_weights = torch.nn.functional.softmax(logits, dim=-1)
        
        output = attn_weights.matmul(self.w_values.reshape([1, 1, -1]))
        return output

# Initializing the model
m = Model(n_feature_keys=32, n_hidden_keys=512)

# Inputs to the model
data = torch.randn(16, 56, 512)  # Data tensor of shape [n_batch, n_feature_values, n_feature_values]
keys = torch.randn(16, 56, 512)  # Key tensor of shape [n_batch, n_feature_values, n_hidden_values]
queries = torch.randn(16, 64, 512)  # Query tensor of shape [n_batch, n_query, n_feature_values]
pad_mask = torch.ones([16, 56, 1])
mask = torch.reshape(torch.logical_not(pad_mask), [16, -1, 1])  # The mask to exclude padding
maskvalue = sys.float_info.min * torch.ones([1])  # The value to fill when the mask if True
