bool IndVarSimplify::run(Loop *L) {
  // We need (and expect!) the incoming loop to be in LCSSA.
  assert(L->isRecursivelyLCSSAForm(*DT, *LI) &&
         "LCSSA required to run indvars!");

  // If LoopSimplify form is not available, stay out of trouble. Some notes:
  //  - LSR currently only supports LoopSimplify-form loops. Indvars'
  //    canonicalization can be a pessimization without LSR to "clean up"
  //    afterwards.
  //  - We depend on having a preheader; in particular,
  //    Loop::getCanonicalInductionVariable only supports loops with preheaders,
  //    and we're in trouble if we can't find the induction variable even when
  //    we've manually inserted one.
  //  - LFTR relies on having a single backedge.
  if (!L->isLoopSimplifyForm())
    return false;

  // Eliminate redundant IV users.
  //
  // Simplification works best when run before other consumers of SCEV. We
  // attempt to avoid evaluating SCEVs for sign/zero extend operations until
  // other expressions involving loop IVs have been evaluated. This helps SCEV
  // set no-wrap flags before normalizing sign/zero extension.
  Rewriter.disableCanonicalMode();
  Changed |= simplifyAndExtend(L, Rewriter, LI);

  return Changed;
}

bool IndVarSimplify::simplifyAndExtend(Loop *L,
                                       SCEVExpander &Rewriter,
                                       LoopInfo *LI) {
  SmallVector<WideIVInfo, 8> WideIVs;

  // Each round of simplification iterates through the SimplifyIVUsers worklist
  // for all current phis, then determines whether any IVs can be
  // widened. Widening adds new phis to LoopPhis, inducing another round of
  // simplification on the wide IVs.
  bool Changed = false;
  while (!LoopPhis.empty()) {
    // Evaluate as many IV expressions as possible before widening any IVs. This
    // forces SCEV to set no-wrap flags before evaluating sign/zero
    // extension. The first time SCEV attempts to normalize sign/zero extension,
    // the result becomes final. So for the most predictable results, we delay
    // evaluation of sign/zero extend evaluation until needed, and avoid running
    // other SCEV based analysis prior to simplifyAndExtend.
    do {
      PHINode *CurrIV = LoopPhis.pop_back_val();      // Information about sign/zero extensions of CurrIV.
      IndVarSimplifyVisitor Visitor(CurrIV, SE, TTI, DT);

      Changed |= simplifyUsersOfIV(CurrIV, SE, DT, LI, TTI, DeadInsts, Rewriter,
                                   &Visitor);

      if (Visitor.WI.WidestNativeType) {
        WideIVs.push_back(Visitor.WI);
      }
    } while(!LoopPhis.empty());

    // Continue if we disallowed widening.
    if (!WidenIndVars)
      continue;

    for (; !WideIVs.empty(); WideIVs.pop_back()) {
      unsigned ElimExt;
      unsigned Widened;
      if (PHINode *WidePhi = createWideIV(WideIVs.back(), LI, SE, Rewriter,
                                          DT, DeadInsts, ElimExt, Widened,
                                          HasGuards, UsePostIncrementRanges)) {
        NumElimExt += ElimExt;
        NumWidened += Widened;
        Changed = true;
        LoopPhis.push_back(WidePhi);
      }
    }
  }
  return Changed;
}

