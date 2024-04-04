bool AlwaysInlineImpl(
    Module &M, bool InsertLifetime, ProfileSummaryInfo &PSI,
    function_ref<AssumptionCache &(Function &)> GetAssumptionCache,
    function_ref<AAResults &(Function &)> GetAAR,
    function_ref<BlockFrequencyInfo &(Function &)> GetBFI) {
  SmallSetVector<CallBase *, 16> Calls;
  bool Changed = false;
  SmallVector<Function *, 16> InlinedFunctions;
  for (Function &F : M) {
    // When callee coroutine function is inlined into caller coroutine function
    // before coro-split pass,
    // coro-early pass can not handle this quiet well.
    // So we won't inline the coroutine function if it have not been unsplited
    if (F.isPresplitCoroutine())
      continue;

  return Changed;
}

PreservedAnalyses AlwaysInlinerPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetAssumptionCache = [&](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto GetBFI = [&](Function &F) -> BlockFrequencyInfo & {
    return FAM.getResult<BlockFrequencyAnalysis>(F);
  };
  auto GetAAR = [&](Function &F) -> AAResults & {
    return FAM.getResult<AAManager>(F);
  };
  auto &PSI = MAM.getResult<ProfileSummaryAnalysis>(M);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

