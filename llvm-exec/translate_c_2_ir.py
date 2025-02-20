### Please generate a valid C Code that meets the requirements below. The code should contain a `main` function. And the main function gives back an output value. Please initialize all the variables you define with a value. Please do not include any undefined behavior in your code.

# Description of requirements

The C code should contain the following pattern:
```
// no va_arg in stdarg.h is used

// define one function which has variable arguments
static int varargs_func(int x, ...) {
    RANDOM_CODE; // Do something here with input arguments, but don't use va_args
    int result = x OPERATOR ANY_VALUE; // result is related to x
    return result;
}

int main(void) {
    RANDOM_CODE;  // declare some variables
    int x = ANY_VALUE; // declare variable x
    ANY_TYPE y = ANY_VALUE; // declare y
    ANY_TYPE z = ANY_VALUE; // declare z
    // invoke the func varargs_func
    int result = varargs_func(x, y, z, ANY_VALUE, ANY_VALUE); // you can pass any number/type of arguments
    return result;
}
```
This pattern characterizes scenarios where within the `main` function, there exists a call to `varargs_func`, and the result of this call is being used. `varargs_func` is marked as variable function, which accepts variable arguments, then returns a single `ANY_TYPE` value that is determined by the first argument. The first argument can be any type, e.g. `int`, `float`, `ptr` `struct`, and so on. The functions must be `static`(which will be translated into ir function with internal attribute), no vastart intrinsic(such as `va_arg` in `stdarg.h`). The `main` function must return a value related to this optimization for further comparison.

# C Code begins 

```c
static int varargs_func(int x, ...) {
    int result = 10;
    if (x % 2) {
        result = x * 32 + 15;
    } else {
        result = x - 15;
    }
    if (x > 100) {
        result = result * 2;
    } else {
        result = result * 2;
    }
    return result;
}

int main(void) {
    int result = 0;
    int a = (1 << 31);
    int arr[4] = {0, 1, 2, 3};
    char str[] = "hello";
    int b = varargs_func(a, 18, 25, arr[1], 1.0, str[3]);
    result = b;
    return result;
}
```
# C Code ends

### Please generate a valid C Code that meets the requirements below. The code should contain a `main` function. And the main function gives back an output value. Please initialize all the variables you define with a value.

# Description of requirements

{Description}

# C Code begins

```
