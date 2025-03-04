// Template for instruction combining
int function_with_combinable_instructions(int x) {
    int temp = x + 5; // Could be combined with the previous instruction
    int result = temp * 2;
    
    return result;
}

int main() {
    int x = 10;
    int result = function_with_combinable_instructions(x);
    return result;
}