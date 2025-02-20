### Please describe the type of C program that triggers the optimization shown in the code. The description should be concise and clear. Use code to illustrate patterns or constraints as needed. Please only describe the characteristics of the program. Do not describe the optimization pass code or what happens after the optimization is triggered.

# Code of the pass
void ADCEPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}

bool ADCEPass::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  
  return AggressiveDeadCodeElimination(F, &AC, &DT, &TLI).performDeadCodeElimination();
}

# Description
The C program should contain the following pattern:
```
int foo() {
int x = 5;    // Dead store
int y = 10;   // Dead store
return 42;    // Only this is used
}
```
This pattern characterizes scenarios where variables are assigned values but never used before the function returns or before being reassigned. The program contains dead code that can be safely eliminated without affecting the program's observable behavior.

### Please describe the type of C program that {}. The description should be concise and clear. Use code to illustrate patterns or constraints as needed. Please only describe the characteristics of the program. Do not describe the optimization pass code or what happens after the optimization is triggered.

{}

# Description