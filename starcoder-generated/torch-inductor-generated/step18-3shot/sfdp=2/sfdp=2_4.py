
class Model(torch.nn.Module):
    def __init__(self, input_size, num_heads, dropout=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = input_size // num_heads
        self.scale_factor = torch.tensor(self.head_size).pow(-0.5)

        self.query = torch.nn.Linear(input_size, input_size)
        self.key = torch.nn.Linear(input_size, input_size)
        self.value = torch.nn.Linear(input_size, input_size)

        self.dropout_p = torch.nn.Parameter(torch.scalar_tensor(dropout))

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

    def generate_config(self):
        self.config = dict(
            input_size=input_size,
            num_heads=num_heads,
            dropout_p=dropout_p.item(),
        )

# Initializing the model
m = Model(input_size=256, num_heads=8)

# Inputs to the model
x = torch.ones(32, 2304, 256) # (batch_size, 2304, 256)
