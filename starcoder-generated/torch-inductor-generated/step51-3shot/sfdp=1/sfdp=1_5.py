
class Model(torch.nn.Module):
    def __init__(self, num_heads=8, dropout_p=0.1, scale_factor=128**-.5):
        super().__init__()
        self.scale_factor = scale_factor
        self.q = torch.nn.Linear(128, 128)
        self.k = torch.nn.Linear(128, 128)
        self.v = torch.nn.Linear(128, 128)

    def forward(self, data, mask):
        q_data = self.q(data)
        k_data = self.k(data)
        v_data = self.v(data)
        qk_data = torch.matmul(q_data, k_data.transpose(-2, -1))
        scaled_qk_data = qk_data.div(self.scale_factor)
        softmax_qk_data = scaled_qk_data.softmax(dim=-1)
        dropout_qk_data = torch.nn.functional.dropout(softmax_qk_data, p=dropout_p, mask=mask)
        output = torch.matmul(dropout_qk_data, v_data)
        return output

# Initializing the model
m = Model()

# Inputs and mask to the model
x1 = torch.randn(1, 64, 128)
mask = torch.zeros(1, 64, 64).bool()
