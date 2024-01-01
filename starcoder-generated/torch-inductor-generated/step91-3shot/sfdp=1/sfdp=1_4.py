
def scaled_dot_product_attention(query, key, value, inv_scale_factor, dropout_p):
    qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_qk = qk.div(inv_scale_factor)
    softmax_qk = scaled_qk.softmax(dim=-1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
    output = dropout_qk.matmul(value)
    return output

class CustomAttentionLayer(torch.nn.Module):
    def __init__(self, dim, heads=1, dropout=0.0, bias=True):
        super().__init__()
        self.proj1 = torch.nn.Linear(dim, dim)
        self.proj2 = torch.nn.Linear(dim, dim, bias=False)
        self.proj3 = torch.nn.Linear(dim, dim, bias=bias)
        self.projection = torch.nn.Linear(dim, dim)
        self.scale_factor = dim ** -0.5
        assert(dim % heads == 0)
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, input, mask=None, return_att_weights=False):
        n = input.size(1)
        a, b, c = input.size(-3), input.size(-2), input.size(-1)
        input = input.view(n, a * b, c).transpose(-2, -1)
        q = self.proj1(input)
        k = self.proj2(input)
        v = self.proj3(input)
        q = q.view(n, a, b, self.heads * c // self.heads)
        k = k.view(n, a, b, self.heads * c // self.heads)
        v = v.view(n, a, b, self.heads * c // self.heads)
        v = v.transpose(-3, -2)
        dropout_qk = scaled_dot_product_attention(q, k, v, self.scale_factor, self.dropout.p)
        dropout_qk = dropout_qk.transpose(-3, -2)
        output = dropout_qk.contiguous().view(n, a * b, self.heads * c // self.heads)
        output = output + input
        output = self.projection(output)
        if return_att_weights:
            att_weights = torch.matmul(q.transpose(2, 3), k.transpose(2, 3)).transpose(1, 2)
            return output, att_weights[:, :a * b // self.heads, :a * b // self.heads]
        return output
 
class Model(torch.nn.Module):
    def __init__(self, dim, depth=2, heads=1, dim_head=None, scale_head=1, dropout=0.0, activation=None, num_classes=100):
        super().__init__()
        dim_head = dim_head or dim
        assert dim_head % heads == 0
        self.heads = heads
        inner_dim = heads * dim_head
        self.scale_factor = scale_head ** -0.5
        padding = (depth % 2 == 0) * (depth // 2) * (dim_head // 2)
        layers = [torch.nn.ModuleList([
            CustomAttentionLayer(dim=dim, heads=heads, dropout=dropout, bias=True),
            torch.nn.Linear(num_classes, num_classes),
        ])]
        for i in range(1, depth):
            layers[-1].append(torch.nn.ModuleList([
                CustomAttentionLayer(dim=dim, heads=heads, dropout=dropout, bias=False),
                torch.nn.Linear(num_classes, num_classes, bias=False),
            ]))
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(num_classes, dim, bias=False),
                torch.nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=padding, bias=False),
            ) for _ in range(depth - 1)
        ]) + layers
        self.activation = activation or torch.nn.ReLU()
        self.output = torch.nn.Linear(num_classes, num_classes)
 
    def forward(self, x1, mask=None, return_att_weights=False):
        n = x1.size(1)
        x = x1
        for i in range(len(self.layers)):
            layer_input = self.layers[i][0](x)
            for j in range(i):
                if j % 2 == 0:
                    layer_input = self.layers[j][0](layer_input, return_att_weights=return_att_weights)
                else:
                    layer_input = self.layers[j][1](layer_input) + layer_input
            if i < len(self.layers) - 1:
                layer_output = self.activation(layer_input)
                for k in range(i - len(self.layers) + 2, 0, -2):
                    if k - 1 < 0:
                        layer_output = self.layers[k - 1][0](layer_output)
                    else:
                        layer_output = self.layers[k - 1][0](layer_output, return_att_weights=return_att_weights)
                    layer_output = layer_output + self.layers[k - 1][1](layer_output)
                    layer_output = self.activation(layer_output)
                x = self.layers[i][1](layer_output)
        output = self.output(x)
        if return_att_weights:
            att_weights = torch.tensor([
                torch.sum(
                    self.scale_factor * torch.sigmoid(t.narrow(-2, 0, self.heads * n // self.heads).contiguous().view(n, self.heads, n // self.heads, n // self.heads)), dim=[-1, -2]
                ).t()
                for t in x1  # torch.stack([torch.sum(self.scale_factor * torch.sigmoid(t.narrow(-2, 0, self.heads * n // self.heads).contiguous().view(n, self.heads, n // self.heads, n // self.heads)), dim=[-1, -2]).t() for t in x1])
            ])
            return output, att_weights[:, :n // self.heads, :n // self.heads]
        return output

# Initializing the model
m = Model(2)

# Inputs to the model
x1 = torch.randn(1, 100, 2)
x2 = torch.randn(100, 100)
