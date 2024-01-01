
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.2, d_model=1, num_head=1, batch=1, seq_len=128):
        super().__init__()
        self.scale = d_model ** -0.5
        self.input_dropout = torch.nn.Dropout(dropout_p)
        self.output_dropout = torch.nn.Dropout(dropout_p)
        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.seq_len = seq_len
        self.batch = batch
        self.num_head = num_head
        self.d_model = d_model

    def forward(self, x1):
        _x1 = x1.permute(1, 0, 2)
        _x1 = _x1.view([_x1.shape[0], -1])
        _x1 = self.input_dropout(_x1)
        _x1 = self.qkv_proj(_x1)
        q,k,v = torch.split(_x1, [_x1.shape[1]//3]*3, dim=0)
        q = q.view(3, self.num_head, self.batch, -1).permute([2, 0, 1, 3])
        k = k.view(3, self.num_head, self.batch, -1).permute([2, 0, 1, 3])
        v = v.view(3, self.num_head, self.batch, -1).permute([2, 0, 1, 3])
        q = q[0] * 0.5
        k = k[0] * 0.5
        v = v[0] * 0.5
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v)
        output = output.transpose(0, 1)
        output = output.reshape(self.num_head, self.batch, -1)
        output = output[1].permute(1, 0, 2)
        output = self.output_dropout(output)
        output = torch.flatten(output, end_dim=1, start_dim=1)
        output = torch.flatten(output)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 1, 256)
