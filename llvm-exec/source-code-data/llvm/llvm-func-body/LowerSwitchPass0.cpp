PreservedAnalyses LowerSwitchPass::run(Function &F,
                                       FunctionAnalysisManager &AM) {
  LazyValueInfo *LVI = &AM.getResult<LazyValueAnalysis>(F);
  AssumptionCache *AC = AM.getCachedResult<AssumptionAnalysis>(F);
  return LowerSwitch(F, LVI, AC) ? PreservedAnalyses::none()
                                 : PreservedAnalyses::all();
}

bool LowerSwitch(Function &F, LazyValueInfo *LVI, AssumptionCache *AC) {
  bool Changed = false;
  SmallPtrSet<BasicBlock *, 8> DeleteList;

    if (SwitchInst *SI = dyn_cast<SwitchInst>(Cur.getTerminator())) {
      Changed = true;
      ProcessSwitchInst(SI, DeleteList, AC, LVI);
    }
  }

  return Changed;
}

