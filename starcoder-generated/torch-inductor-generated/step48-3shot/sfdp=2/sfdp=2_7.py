
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout_p=0, device=torch.device("cpu")):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.head_dim = d_model // num_heads

        # Parameters for the dot product of the query and the key
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)

        self.scaled_dot_product = ScaledDotProductAttention(dropout_p, device)

    def forward(self, x1):
        batch_size, sequence_length, d_model = x1.shape
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
        output = self.scaled_dot_product(q, k, v)
        return output

# Initializing the model
m = Model(512, 8)

# Inputs to the model
x1 = torch.randn(1, 512, 16)
