
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout_p = 0):
        super().__init__()
        self.hidden_size = output_size
        self.project_q = torch.nn.Linear(input_size, output_size, bias = False)
        self.project_k = torch.nn.Linear(input_size, output_size, bias = False)
        self.project_v = torch.nn.Linear(input_size, output_size, bias = False)
        self.drop = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, inv_scale_factor):
        query = self.project_q(query)
        key = self.project_k(key).transpose(-2, -1)
        value = self.project_v(value)
        scaled_qk = torch.matmul(query, key).div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim = -1)
        # Dropout's implementation is inconsistent between PyTorch 1.8.1+cpu and 1.10.1+cu111
        # Here, we convert it to a compatible implementation manually
        # We cannot use torch.nn.functional.dropout() without disabling dropout in PyTorch
        masked_qk = softmax_qk / (1 - self.drop.p)
        output = torch.matmul(masked_qk, value)
        return output

# Initializing the model with dropout probability 0.5
m = Model(100, 50, 0.5)

# Inputs to the model
query = torch.randn(50, 100)
key = torch.randn(90, 100)
value = torch.randn(90, 100)
inv_scale_factor = math.sqrt(float(0.5))
