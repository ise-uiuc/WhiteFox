 
import torch

class Model(torch.nn.Module):
    def __init__(self, query_shape: Tuple[int,...], key_shape: Tuple[int,...], value_shape: Tuple[int,...], dropout_p: float):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = lambda input: torch.nn.functional.softmax(input, dim=-1)
        self.matmul = lambda input1, input2: input1.matmul(input2.transpose(-2, -1))
        self.div = lambda input1, input2: input1.div(input2)
        self.matmul_div_dropout = lambda input1, input2, input3: self.dropout(self.softmax(self.div(self.matmul(input1, input2), input3)))
        self.matmul3 = lambda input1, input2, input3: self.matmul_div_dropout(input1, input2, input3).matmul(input3)
        self.expand = lambda input: torch.repeat_interleave(input, 1, dim=1)
        self.unsqueeze = lambda input: input.unsqueeze(dim=1)
        self.unsqueeze2 = lambda input: input.unsqueeze(dim=-2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        q = torch.reshape(query, (-1, query.shape[-1]))
        k = torch.reshape(key, (-1, key.shape[-1]))
        v = torch.reshape(value, (-1, value.shape[-1]))
        qk = self.div(self.matmul(q, self.unsqueeze2(k)), math.sqrt(k.shape[-1]))
        softmax_qk = self.softmax(qk)
        dropout_qk = self.dropout(softmax_qk)
        dropout_val = self.dropout(v)
        result = self.matmul3(self.unsqueeze2(self.expand(dropout_qk)), dropout_val, self.unsqueeze(dropout_qk))
        out = torch.reshape(result, (-1, result.shape[1], *result.shape[2:]))
        return out

# Initializing the model
m = Model((2, 10), (1, 10), (1, 10), dropout_p=0.1)

# Inputs to the model
query = torch.randn(2, 10)
key = torch.randn(1, 10)
value = torch.randn(1, 10)
