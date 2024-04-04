PreservedAnalyses StripSymbolsPass::run(Module &M, ModuleAnalysisManager &AM) {
  StripDebugInfo(M);
  StripSymbolNames(M, false);
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

static bool StripSymbolNames(Module &M, bool PreserveDbgInfo) {

  // Remove all names from types.
  StripTypeNames(M, PreserveDbgInfo);

  return true;
}

