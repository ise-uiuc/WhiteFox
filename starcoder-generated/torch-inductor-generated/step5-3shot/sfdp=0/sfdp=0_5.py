
class Model(torch.nn.Module):
    def __init__(self, feature_dim=256):
        super(Model, self).__init__()
        self.dim_per_head = feature_dim // 2 # Set the dimension per head to 128
        # Initialize two Linear layers to transform key and query vectors
        # (with the same number of features per vector)
        self.linear_key = torch.nn.Linear(feature_dim, feature_dim)
        self.linear_query = torch.nn.Linear(feature_dim, feature_dim)
        self.linear_value = torch.nn.Linear(feature_dim, feature_dim)
        # Initialize one layer for the output of a single attention head
        self.linear_final = torch.nn.Linear(self.dim_per_head, 256)

    def forward(self, x1, x2, x3):
        # Transform the key, query and value tensors with the Linear layers
        k = self.linear_key(x1)
        q = self.linear_query(x2)
        v = self.linear_value(x3)
        # The shape of the key/query/value tensors will change from
        #   (batch_size, feature_size, seq_length, feature_dim)
        # to (batch_size, feature_dim, seq_length, feature_dim)
        k = torch.cat(k.split(self.dim_per_head, dim=-1), dim=0)
        q = torch.cat(q.split(self.dim_per_head, dim=-1), dim=0)
        v = torch.cat(v.split(self.dim_per_head, dim=-1), dim=0)
        # Compute the scaled dot-product attention weights
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1))
        scaled_dot_product = scaled_dot_product / np.sqrt(self.dim_per_head)
        scaled_dot_product = torch.softmax(scaled_dot_product, dim=-1)
        # Compute the attention heads
        output = scaled_dot_product.matmul(v)
        # Reshape the attention heads and concatenate
        output = torch.cat(output.split(1, dim=0), dim=-1)
        # Compute the output of a single attention head
        m = self.linear_final(output)
        return m
```

# Initializing the model
m = Model(feature_dim=256)

# Inputs to the model
x1 = torch.randn(4, 64, 256)
x2 = torch.randn(5, 64, 256)
x2 = torch.randn(5, 64, 256)
