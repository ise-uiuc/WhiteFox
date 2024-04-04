PreservedAnalyses IPSCCPPass::run(Module &M, ModuleAnalysisManager &AM) {
  const DataLayout &DL = M.getDataLayout();
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> const TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  auto GetTTI = [&FAM](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };
  auto GetAC = [&FAM](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto getAnalysis = [&FAM, this](Function &F) -> AnalysisResultsForFn {
    DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    return {
        std::make_unique<PredicateInfo>(F, DT, FAM.getResult<AssumptionAnalysis>(F)),
        &DT, FAM.getCachedResult<PostDominatorTreeAnalysis>(F),
        isFuncSpecEnabled() ? &FAM.getResult<LoopAnalysis>(F) : nullptr };
  };

  if (!runIPSCCP(M, DL, &FAM, GetTLI, GetTTI, GetAC, getAnalysis,
                 isFuncSpecEnabled()))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<FunctionAnalysisManagerModuleProxy>();
  return PA;
}

static bool runIPSCCP(
    Module &M, const DataLayout &DL, FunctionAnalysisManager *FAM,
    std::function<const TargetLibraryInfo &(Function &)> GetTLI,
    std::function<TargetTransformInfo &(Function &)> GetTTI,
    std::function<AssumptionCache &(Function &)> GetAC,
    function_ref<AnalysisResultsForFn(Function &)> getAnalysis,
    bool IsFuncSpecEnabled) {
  SCCPSolver Solver(DL, GetTLI, M.getContext());
  FunctionSpecializer Specializer(Solver, M, FAM, GetTLI, GetTTI, GetAC);

  // If we inferred constant or undef values for globals variables, we can
  // delete the global and any stores that remain to it.
  for (const auto &I : make_early_inc_range(Solver.getTrackedGlobals())) {
    GlobalVariable *GV = I.first;
    if (SCCPSolver::isOverdefined(I.second))
      continue;
    LLVM_DEBUG(dbgs() << "Found that GV '" << GV->getName()
                      << "' is constant!\n");
    while (!GV->use_empty()) {
      StoreInst *SI = cast<StoreInst>(GV->user_back());
      SI->eraseFromParent();
    }
    MadeChanges = true;
    M.eraseGlobalVariable(GV);
    ++NumGlobalConst;
  }

  return MadeChanges;
}

