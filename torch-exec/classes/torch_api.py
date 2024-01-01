import torch
from classes.argdef import ArgDef
from classes.argument import *
from classes.api import *
from utils.probability import (
    add_tensor_dimension,
    rm_tensor_dimension,
    change_tensor_dimension,
    change_tensor_shape,
    change_tensor_dtype,
)
import json
import random


class TorchArgument(Argument):
    _supported_types = [
        ArgType.TORCH_DTYPE,
        ArgType.TORCH_OBJECT,
        ArgType.TORCH_TENSOR,
    ]
    _dtypes = [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.complex64,
        torch.complex128,
        torch.bool,
    ]
    _memory_formats = [
        torch.contiguous_format,
        torch.channels_last,
        # torch.preserve_format,
    ]
    _layouts = [
        torch.strided,
        torch.sparse_coo,
        torch.sparse_bsc,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
    ]
    _float_complex_dtypes = [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.complex64,
        torch.complex128,
    ]
    _min_values = [0] + [-(1 << i) for i in range(0, 8)]
    _max_values = [(1 << i) - 1 for i in range(0, 8)]
    _tensor_size_limit = 128
    _value_limit = 128

    def __init__(
        self,
        value,
        type: ArgType,
        shape=None,
        dtype=None,
        max_value=1,
        min_value=0,
        memory_format=torch.contiguous_format,
    ):
        super().__init__(value, type)
        self.shape = shape
        self.dtype = dtype
        self.max_value = max_value
        self.min_value = min_value
        self.grad = False
        self.new_tensor = True
        self.memory_format = memory_format
        self.layout = torch.strided

    def to_code(
        self,
        var_name,
        device="cpu",
        low_precision=False,
        is_sparse=False,
        use_old_tensor=False,
    ) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                self.value[i].grad = self.grad
                self.value[i].new_tensor = self.new_tensor
                code += self.value[i].to_code(
                    f"{var_name}_{i}",
                    device=device,
                    low_precision=low_precision,
                    use_old_tensor=use_old_tensor,
                )
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TORCH_TENSOR:
            dtype = self.dtype
            max_value = self.max_value
            min_value = self.min_value
            memory_format = self.memory_format
            if low_precision:
                dtype = self.low_precision_dtype(dtype)
                max_value, min_value = self.random_tensor_value(dtype)

            # FIXME: shape tune
            shape = self.shape
            size = 1
            for i in range(len(shape) - 1, -1, -1):
                if size * shape[i] > TorchArgument._tensor_size_limit:
                    shape[i] = 1
                else:
                    size *= shape[i]

            suffix = ""
            if is_sparse:
                suffix += ".to_sparse()"
            if self.grad and self.is_float_or_complex_tensor():
                suffix += ".requires_grad_()"

            code = ""
            if self.new_tensor and not use_old_tensor:
                if memory_format == torch.channels_last:
                    if len(shape) == 5:
                        memory_format = torch.channels_last_3d
                    elif len(shape) != 4:
                        memory_format = torch.contiguous_format

                default_arg = f"{shape}, dtype={dtype}, memory_format={memory_format}"
                code += f"{var_name}_tensor = torch.empty({default_arg})\n"

                if dtype.is_floating_point:
                    code += f"{var_name}_tensor.uniform_({min_value}, {max_value})\n"
                elif dtype.is_complex:
                    code += f"{var_name}_tensor.uniform_({min_value}, {max_value})\n"
                else:
                    code += f"{var_name}_tensor = torch.randint_like({var_name}_tensor, {min_value}, {max_value+1})\n"
            code += f"{var_name} = {var_name}_tensor.clone().detach().to('{device}'){suffix}\n"
            return code
        elif self.type == ArgType.TORCH_OBJECT:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.TORCH_DTYPE:
            return f"{var_name} = {self.value}\n"
        return super().to_code(var_name)

    def to_diff_code(self, var_name, oracle):
        """differential testing with oracle"""
        code = ""
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", oracle)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
        elif self.type == ArgType.TORCH_TENSOR:
            if oracle == OracleType.CUDA:
                code += f"{var_name} = {var_name}_tensor.clone().cuda()\n"
            elif oracle == OracleType.PRECISION:
                code += f"{var_name} = {var_name}_tensor.clone().type({self.dtype})\n"
        return code

    def mutate_tensor(self):
        tensor_mutators = {
            self.mutate_tensor_value_range: 0.4,
            self.mutate_tensor_dtype: 0.3,
            self.mutate_tensor_shape: 0.28,
            self.mutate_tensor_format: 0.02,
            # self.mutate_tensor_layout: 0.05
        }
        mutator = choice(list(tensor_mutators.keys()), p=list(tensor_mutators.values()))
        mutator()

    def mutate_tensor_value_range(self):
        self.max_value, self.min_value = self.random_tensor_value(self.dtype)

    def mutate_tensor_dtype(self):
        self.dtype = choice(self._dtypes)
        self.mutate_tensor_value_range()

    def mutate_tensor_shape(self):
        new_size = list(self.shape)
        if add_tensor_dimension():
            new_size = [1] + new_size
        if rm_tensor_dimension() and len(new_size) > 0:
            new_size = new_size[1:]
        # change the shape
        for i in range(len(new_size)):
            if change_tensor_shape():
                new_size[i] = self.mutate_int_value(
                    new_size[i], _min=0, _max=self._value_limit
                )

    def mutate_tensor_format(self):
        self.memory_format = choice(self._memory_formats)

    def mutate_tensor_layout(self):
        self.layout = choice(self._layouts)

    def mutate_value(self, _min=None, _max=None) -> None:
        if self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TORCH_TENSOR:
            self.max_value, self.min_value = self.random_tensor_value(self.dtype)
        elif self.type in super()._support_types:
            # super().mutate_value(_min=_min, _max=_max)
            # TODO: now I set the limit for the int value
            super().mutate_value(_min=-self._value_limit, _max=self._value_limit)
        else:
            print(self.type, self.value)
            assert 0

    def mutate_type(self) -> None:
        if self.type == ArgType.NULL:
            # choose from all types
            new_type = choice(self._support_types + super()._support_types)
            self.type = new_type
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    TorchArgument(2, ArgType.INT),
                    TorchArgument(3, ArgType.INT),
                ]
            elif new_type == ArgType.TORCH_TENSOR:
                self.shape = [2, 2]
                self.dtype = torch.float32
            elif new_type == ArgType.TORCH_DTYPE:
                self.value = choice(self._dtypes)
            elif new_type == ArgType.TORCH_OBJECT:
                self.value = choice(self._memory_formats)
            else:
                self.value = super().initial_value(new_type)
        elif self.type == ArgType.TORCH_TENSOR:
            new_size = list(self.shape)
            # change the dimension of tensor
            if change_tensor_dimension():
                if add_tensor_dimension():
                    new_size.append(1)
                elif len(new_size) > 0:
                    new_size.pop()
            # change the shape
            for i in range(len(new_size)):
                if change_tensor_shape():
                    new_size[i] = self.mutate_int_value(new_size[i], _min=0)
            self.shape = new_size
            # change dtype
            if change_tensor_dtype():
                self.dtype = choice(self._dtypes)
                self.max_value, self.min_value = self.random_tensor_value(self.dtype)
        elif self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type in super()._support_types:
            super().mutate_type()
        else:
            print(self.type, self.value)
            assert 0

    def to_record(self):
        if self.type == ArgType.TORCH_TENSOR:
            shape = []
            for i in self.shape:
                shape.append(int(i))
            record_value = {
                "shape": shape,
                "dtype": str(self.dtype),
                "max_value": self.max_value,
                "min_value": self.min_value,
                "memory_format": str(self.memory_format),
            }
        elif self.type == ArgType.TORCH_DTYPE:
            record_value = str(self.value)
        elif self.type == ArgType.TORCH_OBJECT:
            record_value = str(self.value)
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            temp = []
            for arg in self.value:
                temp.append(arg.to_record())
            record_value = temp
        elif self.type == ArgType.INT:
            record_value = int(self.value)
        else:
            record_value = self.value

        return {
            "type": int(self.type),
            "value": record_value,
        }

    def is_float_or_complex_tensor(self):
        return (
            self.type == ArgType.TORCH_TENSOR
            and self.dtype in self._float_complex_dtypes
        )

    @staticmethod
    def generate_arg_from_record(record):
        type = ArgType(record["type"])
        value = record["value"]

        if type == ArgType.NULL:
            return TorchArgument(None, type)
        elif type in [ArgType.BOOL, ArgType.INT, ArgType.FLOAT, ArgType.STR]:
            return TorchArgument(value, type)
        elif type in [ArgType.TORCH_DTYPE, ArgType.TORCH_OBJECT]:
            return TorchArgument(eval(value), type)
        elif type == ArgType.TORCH_TENSOR:
            shape = value["shape"]
            dtype = eval(value["dtype"])
            max_value = value["max_value"]
            min_value = value["min_value"]
            memory_format = eval(value["memory_format"])
            return TorchArgument(
                None,
                ArgType.TORCH_TENSOR,
                shape,
                dtype=dtype,
                max_value=max_value,
                min_value=min_value,
                memory_format=memory_format,
            )
        elif type in [ArgType.TUPLE, ArgType.LIST]:
            res = []
            for r in value:
                res.append(TorchArgument.generate_arg_from_record(r))
            return TorchArgument(res, type)
        else:
            assert 0

    @staticmethod
    def random_tensor_value(dtype):
        max_value = 1
        min_value = 0
        if dtype == torch.bool:
            min_value = choice([0, 1])
            max_value = max(min_value, choice([0, 1]))
        elif dtype == torch.uint8:
            min_value = 0
            max_value = choice(TorchArgument._max_values)
        else:
            min_value = choice(TorchArgument._min_values)
            max_value = choice(TorchArgument._max_values)
        return max_value, min_value

    @staticmethod
    def generate_arg_from_signature(signature):
        """Generate a Torch argument from the signature"""
        if signature == "torchTensor":
            return TorchArgument(
                None, ArgType.TORCH_TENSOR, shape=[2, 2], dtype=torch.float32
            )
        if signature == "torchdtype":
            return TorchArgument(choice(TorchArgument._dtypes), ArgType.TORCH_DTYPE)
        if isinstance(signature, str) and signature == "torchdevice":
            value = torch.device("cpu")
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torchmemory_format":
            value = choice(TorchArgument._memory_formats)
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torch.strided":
            return TorchArgument("torch.strided", ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature.startswith("torch."):
            value = eval(signature)
            if isinstance(value, torch.dtype):
                return TorchArgument(value, ArgType.TORCH_DTYPE)
            elif isinstance(value, torch.memory_format):
                return TorchArgument(value, ArgType.TORCH_OBJECT)
            print(signature)
            # TODO:
            assert 0
        if isinstance(signature, bool):
            return TorchArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TorchArgument(signature, ArgType.INT)
        if isinstance(signature, str):
            return TorchArgument(signature, ArgType.STR)
        if isinstance(signature, float):
            return TorchArgument(signature, ArgType.FLOAT)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.TUPLE)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.LIST)
        # signature is a dictionary
        if isinstance(signature, dict):
            if not ("shape" in signature.keys() and "dtype" in signature.keys()):
                raise Exception("Wrong signature {0}".format(signature))
            shape = signature["shape"]
            dtype = signature["dtype"]

            if "max_value" in signature and "min_value" in signature:
                max_value = signature["max_value"]
                min_value = signature["min_value"]
            else:
                max_value, min_value = TorchArgument.random_tensor_value(dtype)

            if "memeory_format" in signature:
                memory_format = signature["memory_format"]
            else:
                memory_format = torch.contiguous_format

            # signature is a ndarray or tensor.
            if isinstance(shape, (list, tuple)):
                if not dtype.startswith("torch."):
                    dtype = f"torch.{dtype}"
                dtype = eval(dtype)
                return TorchArgument(
                    None,
                    ArgType.TORCH_TENSOR,
                    shape,
                    dtype=dtype,
                    max_value=max_value,
                    min_value=min_value,
                    memory_format=memory_format,
                )
            else:
                return TorchArgument(
                    None,
                    ArgType.TORCH_TENSOR,
                    shape=[2, 2],
                    dtype=torch.float32,
                )
        return TorchArgument(None, ArgType.NULL)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [torch.int16, torch.int32, torch.int64]:
            return torch.int8
        elif dtype in [torch.float32, torch.float64]:
            return torch.float16
        elif dtype in [torch.complex64, torch.complex128]:
            return torch.complex32
        return dtype

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res is not None:
            return res
        if isinstance(x, torch.Tensor):
            return ArgType.TORCH_TENSOR
        elif isinstance(x, torch.dtype):
            return ArgType.TORCH_DTYPE
        else:
            return ArgType.TORCH_OBJECT


class TorchAPI(API):
    # indices of sparse tensor for sparse API
    _sparse_API = {
        "torch.sspaddmm": [0, 1],
        "torch.Tensor.sspaddmm": [0, 1],
        "torch.sparse.sum": [0],
        "torch.sparse.addmm": [1],
        "torch.sparse.mm": [0],
        "torch.hspmm": [0],
        "torch.smm": [0],
        "torch.Tensor.smm": [0],
        "torch.sparse.softmax": [0],
        "torch.sparse.log_softmax": [0],
    }

    def __init__(self, api_name):
        super().__init__(api_name)
        self.is_class = inspect.isclass(eval(self.api))
