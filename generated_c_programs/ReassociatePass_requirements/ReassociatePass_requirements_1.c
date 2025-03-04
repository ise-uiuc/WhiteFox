// Template for reassociation
int function_with_reassociable_ops(int a, int b, int c) {
    // Operations that could be reassociated for better optimization
    return (a + b) + c;
}

int main() {
    int x = 10;
    int y = 20;
    int z = 30;
    int result = function_with_reassociable_ops(x, y, z);
    return result;
}