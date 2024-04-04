static bool runImpl(Function &F, AssumptionCache &AC, TargetTransformInfo &TTI,
                    TargetLibraryInfo &TLI, DominatorTree &DT,
                    AliasAnalysis &AA) {
  bool MadeChange = false;
  const DataLayout &DL = F.getParent()->getDataLayout();
  TruncInstCombine TIC(AC, TLI, DL, DT);
  MadeChange |= TIC.run(F);
  MadeChange |= foldUnusualPatterns(F, DT, TTI, TLI, AA);
  return MadeChange;
}

bool TruncInstCombine::run(Function &F) {
  bool MadeIRChange = false;

    if (Type *NewDstSclTy = getBestTruncatedType()) {
      LLVM_DEBUG(
          dbgs() << "ICE: TruncInstCombine reducing type of expression graph "
                    "dominated by: "
                 << CurrentTruncInst << '\n');
      ReduceExpressionGraph(NewDstSclTy);
      ++NumExprsReduced;
      MadeIRChange = true;
    }
  }

  return MadeIRChange;
}

