Instruction *InstCombinerImpl::visitTrunc(TruncInst &Trunc) {
  if (Instruction *Result = commonCastTransforms(Trunc))
    return Result;

    // If this cast is a truncate, evaluting in a different type always
    // eliminates the cast, so it is always a win.
    LLVM_DEBUG(
        dbgs() << "ICE: EvaluateInDifferentType converting expression type"
                  " to avoid cast: "
               << Trunc << '\n');
    Value *Res = EvaluateInDifferentType(Src, DestTy, false);
    assert(Res->getType() == DestTy);
    return replaceInstUsesWith(Trunc, Res);
  }

  // Test if the trunc is the user of a select which is part of a
  // minimum or maximum operation. If so, don't do any more simplification.
  // Even simplifying demanded bits can break the canonical form of a
  // min/max.
  Value *LHS, *RHS;
  if (SelectInst *Sel = dyn_cast<SelectInst>(Src))
    if (matchSelectPattern(Sel, LHS, RHS).Flavor != SPF_UNKNOWN)
      return nullptr;

  return nullptr;
}

