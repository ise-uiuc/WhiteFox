PreservedAnalyses LowerGlobalDtorsPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  bool Changed = runImpl(M);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

static bool runImpl(Module &M) {

  GlobalVariable *GV = M.getGlobalVariable("llvm.global_dtors");
  if (!GV || !GV->hasInitializer())
    return false;

  const ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!InitList)
    return false;

  // Validate @llvm.global_dtor's type.
  auto *ETy = dyn_cast<StructType>(InitList->getType()->getElementType());
  if (!ETy || ETy->getNumElements() != 3 ||
      !ETy->getTypeAtIndex(0U)->isIntegerTy() ||
      !ETy->getTypeAtIndex(1U)->isPointerTy() ||
      !ETy->getTypeAtIndex(2U)->isPointerTy())
    return false; // Not (int, ptr, ptr).

  // Collect the contents of @llvm.global_dtors, ordered by priority. Within a
  // priority, sequences of destructors with the same associated object are
  // recorded so that we can register them as a group.
  std::map<
      uint16_t,
      std::vector<std::pair<Constant *, std::vector<Constant *>>>
  > DtorFuncs;
  for (Value *O : InitList->operands()) {
    auto *CS = dyn_cast<ConstantStruct>(O);
    if (!CS)
      continue; // Malformed.    auto *Priority = dyn_cast<ConstantInt>(CS->getOperand(0));
    if (!Priority)
      continue; // Malformed.
    uint16_t PriorityValue = Priority->getLimitedValue(UINT16_MAX);

    Constant *DtorFunc = CS->getOperand(1);
    if (DtorFunc->isNullValue())
      break; // Found a null terminator, skip the rest.

    Constant *Associated = CS->getOperand(2);
    Associated = cast<Constant>(Associated->stripPointerCasts());

    auto &AtThisPriority = DtorFuncs[PriorityValue];
    if (AtThisPriority.empty() || AtThisPriority.back().first != Associated) {
        std::vector<Constant *> NewList;
        NewList.push_back(DtorFunc);
        AtThisPriority.push_back(std::make_pair(Associated, NewList));
    } else {
        AtThisPriority.back().second.push_back(DtorFunc);
    }
  }
  if (DtorFuncs.empty())
    return false;

  // Now that we've lowered everything, remove @llvm.global_dtors.
  GV->eraseFromParent();

  return true;
}

