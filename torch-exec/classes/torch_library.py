from classes.torch_api import *
from classes.library import Library
from classes.argument import *
from classes.api import *
from constant.keys import *


class TorchLibrary(Library):
    def __init__(
        self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3
    ) -> None:
        super().__init__(output_dir)
        self.diff_bound = diff_bound
        self.time_bound = time_bound
        self.time_thresold = time_thresold

    @staticmethod
    def is_equal(
        x,
        y,
        atol=1e-1,
        rtol=1e-3,
        equal_nan=False,
        broadcast=False,
        dtype_strict=False,
        device_strict=False,
        sparse_strict=False,
    ):
        try:
            return TorchLibrary.is_equal_unsafe(
                x,
                y,
                atol=atol,
                rtol=rtol,
                equal_nan=equal_nan,
                broadcast=broadcast,
                dtype_strict=dtype_strict,
                device_strict=device_strict,
                sparse_strict=sparse_strict,
            )
        except Exception as e:
            # print(e)
            return False

    @staticmethod
    def is_equal_unsafe(
        x,
        y,
        atol=1e-1,
        rtol=1e-3,
        equal_nan=False,
        broadcast=False,
        dtype_strict=False,
        device_strict=False,
        sparse_strict=False,
    ):
        def eq_float_tensor(x, y):
            # not strictly equal
            return torch.allclose(x, y, atol=atol, rtol=rtol, equal_nan=equal_nan)

        x_type = TorchArgument.get_type(x)
        y_type = TorchArgument.get_type(y)
        if x_type != y_type:
            # convert basic type to tensor
            if x_type in [ArgType.LIST, ArgType.TUPLE] and y_type in [
                ArgType.LIST,
                ArgType.TUPLE,
            ]:
                pass
            else:
                try:
                    x = torch.tensor(x)
                    y = torch.tensor(y)
                    x_type = TorchArgument.get_type(x)
                    y_type = TorchArgument.get_type(y)
                except Exception:
                    return False
        if x_type == ArgType.TORCH_TENSOR:
            if x.shape != y.shape:
                if broadcast:
                    x = x.flatten()
                    y = y.flatten()
                else:
                    return False
            if x.device != y.device and device_strict:
                return False
            # compare the tensor at the device of x
            device = x.device
            x = x.clone().to(device)
            y = y.clone().to(device)
            if x.dtype != y.dtype:
                if dtype_strict:
                    return False
                else:
                    # we need to cast the high-precision one into low-precision
                    promoted_type = torch.promote_types(x.dtype, y.dtype)
                    low_precision_dtype = (
                        x.dtype if x.dtype != promoted_type else y.dtype
                    )
                    x = x.to(dtype=low_precision_dtype)
                    y = y.to(dtype=low_precision_dtype)

            if x.is_sparse != y.is_sparse and sparse_strict:
                return False

            if x.is_sparse:
                x = x.to_dense()
            if y.is_sparse:
                y = y.to_dense()
            if x.is_complex():
                if not y.is_complex():
                    return False
                return eq_float_tensor(x.real, y.real) and eq_float_tensor(
                    x.imag, y.imag
                )
            if not x.dtype.is_floating_point:
                return torch.equal(x.cpu(), y.cpu())
            # print('hi')
            return eq_float_tensor(x, y)
        elif x_type == ArgType.FLOAT:
            return abs(x - y) < atol + rtol * abs(x)
        elif x_type in [ArgType.LIST, ArgType.TUPLE]:
            if len(x) != len(y):
                return False
            for i in range(len(x)):
                if (
                    TorchLibrary.is_equal_unsafe(
                        x[i],
                        y[i],
                        atol,
                        rtol,
                        equal_nan=equal_nan,
                        broadcast=broadcast,
                        dtype_strict=dtype_strict,
                        device_strict=device_strict,
                        sparse_strict=sparse_strict,
                    )
                    is False
                ):
                    return False
            return True
        else:
            return x == y
