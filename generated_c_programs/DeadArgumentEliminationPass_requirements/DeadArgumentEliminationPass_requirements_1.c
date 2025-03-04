// Template for dead argument elimination
static int function_with_dead_args(int x, int y, int z) {
    // Parameter z is never used in the function body
    return x + y;   // Only x and y are used
}

int main() {
    int a = 10;
    int b = 20;
    int c = 30;   // This value is passed but never used in the function
    int result = function_with_dead_args(a, b, c);
    return result;
}