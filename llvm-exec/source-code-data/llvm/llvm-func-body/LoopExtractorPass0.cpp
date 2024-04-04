bool LoopExtractor::extractLoops(Loop::iterator From, Loop::iterator To,
                                 LoopInfo &LI, DominatorTree &DT) {
  bool Changed = false;
  SmallVector<Loop *, 8> Loops;

    Changed |= extractLoop(L, LI, DT);
    if (!NumLoops)
      break;
  }
  return Changed;
}

bool LoopExtractor::extractLoop(Loop *L, LoopInfo &LI, DominatorTree &DT) {

  assert(NumLoops != 0);
  Function &Func = *L->getHeader()->getParent();
  AssumptionCache *AC = LookupAssumptionCache(Func);
  CodeExtractorAnalysisCache CEAC(Func);
  CodeExtractor Extractor(DT, *L, false, nullptr, nullptr, AC);
  if (Extractor.extractCodeRegion(CEAC)) 

