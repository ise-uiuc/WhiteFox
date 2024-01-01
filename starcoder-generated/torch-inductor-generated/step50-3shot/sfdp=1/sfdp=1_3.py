
class Model(torch.nn.Module):
    def __init__(self, input_size: int, num_heads: int):
        super().__init__()
        self.q = torch.nn.Linear(input_size, input_size)
        self.k = torch.nn.Linear(input_size, input_size)
        self.v = torch.nn.Linear(input_size, input_size)
        self.output = torch.nn.Linear(input_size, input_size)
        self.input_size = input_size

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)).div(self.input_size ** 0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = torch.matmul(dropout_qk, v)
        output = self.output(output)
        return output

# Initializing the model
m = Model(1000, 16)

# Inputs to the model
x = torch.randn(1, time_steps, input_size)
