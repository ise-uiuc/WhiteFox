
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        qk = torch.matmul(hidden_states, hidden_states.transpose(-2, -1)) 
        inv_scale_factor = torch.rsqrt(torch.tensor([hidden_states.shape[-1]]))
        dropout_p = 0.1
        scaled_qk = qk.div(inv_scale_factor) 
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) 
        output = dropout_qk.matmul(hidden_states) 
        return output

# Initializing the model
m = Model(hidden_size=64)

# Inputs to the model
hidden_states = torch.randn(1, 128, 64)
