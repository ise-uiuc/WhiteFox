
class Model(torch.nn.Module):

    def __init__(self, n_heads: int = 1, qkv_dim: int = 8, hidden_dim: int = 16, max_len: int = 516):
        super().__init__()
        self.queries = torch.nn.Linear(hidden_dim, qkv_dim * n_heads, bias=False)
        self.keys = torch.nn.Linear(hidden_dim, qkv_dim * n_heads, bias=False)
        self.values = torch.nn.Linear(hidden_dim, qkv_dim * n_heads, bias=False)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, attn_weights: Tensor) -> Tensor:
        qkv = (
            self.queries(queries).reshape(1, queries.size(0), 4, -1),
            self.keys(keys).reshape(1, keys.size(0), 4, -1),
            self.values(values).reshape(1, values.size(0), 4, -1),
        )

        qkv_weight, _, _ = torch.triu_indices(3, 4)
        weights = attn_weights[:, qkv_weight].reshape((1, queries.size(0), 3, 4) - -1, -1).softmax(dim=-1)
        weighted_q, weighted_k, weighted_v = (t[:, :, qkv_weight, :].reshape(
            (1, queries.size(0), 3, 4) + -1) for t in qkv)
        weight_sum_wv = weights * weighted_v
        result = (weight_sum_wv.sum(dim=3)).reshape(1, queries.size(0), -1)
        return result

# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(2, 2, 16)
x2 = torch.randn(2, 4, 16)
x3 = torch.randn(2, 4, 16)
x4 = torch.randn(2, 4, 4)
