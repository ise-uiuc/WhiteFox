Instruction *InstCombinerImpl::foldCmpLoadFromIndexedGlobal(
    LoadInst *LI, GetElementPtrInst *GEP, GlobalVariable *GV, CmpInst &ICI,
    ConstantInt *AndCst) {
  if (LI->isVolatile() || LI->getType() != GEP->getResultElementType() ||
      GV->getValueType() != GEP->getSourceElementType() ||
      !GV->isConstant() || !GV->hasDefinitiveInitializer())
    return nullptr;

  Constant *Init = GV->getInitializer();
  if (!isa<ConstantArray>(Init) && !isa<ConstantDataArray>(Init))
    return nullptr;

  uint64_t ArrayElementCount = Init->getType()->getArrayNumElements();
  // Don't blow up on huge arrays.
  if (ArrayElementCount > MaxArraySizeForCombine)
    return nullptr;

  // There are many forms of this optimization we can handle, for now, just do
  // the simple index into a single-dimensional array.
  //
  // Require: GEP GV, 0, i {{, constant indices}}
  if (GEP->getNumOperands() < 3 ||
      !isa<ConstantInt>(GEP->getOperand(1)) ||
      !cast<ConstantInt>(GEP->getOperand(1))->isZero() ||
      isa<Constant>(GEP->getOperand(2)))
    return nullptr;

  Type *EltTy = Init->getType()->getArrayElementType();
  for (unsigned i = 3, e = GEP->getNumOperands(); i != e; ++i) {
    ConstantInt *Idx = dyn_cast<ConstantInt>(GEP->getOperand(i));
    if (!Idx) return nullptr;  // Variable index.    uint64_t IdxVal = Idx->getZExtValue();
    if ((unsigned)IdxVal != IdxVal) return nullptr; // Too large array index.

    if (StructType *STy = dyn_cast<StructType>(EltTy))
      EltTy = STy->getElementType(IdxVal);
    else if (ArrayType *ATy = dyn_cast<ArrayType>(EltTy)) {
      if (IdxVal >= ATy->getNumElements()) return nullptr;
      EltTy = ATy->getElementType();
    } else {
      return nullptr; // Unknown type.
    }

    LaterIndices.push_back(IdxVal);
  }

  // Scan the array and see if one of our patterns matches.
  Constant *CompareRHS = cast<Constant>(ICI.getOperand(1));
  for (unsigned i = 0, e = ArrayElementCount; i != e; ++i) {
    Constant *Elt = Init->getAggregateElement(i);
    if (!Elt) return nullptr;    // If this is indexing an array of structures, get the structure element.
    if (!LaterIndices.empty()) {
      Elt = ConstantFoldExtractValueInstruction(Elt, LaterIndices);
      if (!Elt)
        return nullptr;
    }

    // If the element is masked, handle it.
    if (AndCst) Elt = ConstantExpr::getAnd(Elt, AndCst);

    // Find out if the comparison would be true or false for the i'th element.
    Constant *C = ConstantFoldCompareInstOperands(ICI.getPredicate(), Elt,
                                                  CompareRHS, DL, &TLI);
    // If the result is undef for this element, ignore it.
    if (isa<UndefValue>(C)) {
      // Extend range state machines to cover this element in case there is an
      // undef in the middle of the range.
      if (TrueRangeEnd == (int)i-1)
        TrueRangeEnd = i;
      if (FalseRangeEnd == (int)i-1)
        FalseRangeEnd = i;
      continue;
    }

    // If we can't compute the result for any of the elements, we have to give
    // up evaluating the entire conditional.
    if (!isa<ConstantInt>(C)) return nullptr;

    // Otherwise, we know if the comparison is true or false for this element,
    // update our state machines.
    bool IsTrueForElt = !cast<ConstantInt>(C)->isZero();

    // State machine for single/double/range index comparison.
    if (IsTrueForElt) {
      // Update the TrueElement state machine.
      if (FirstTrueElement == Undefined)
        FirstTrueElement = TrueRangeEnd = i;  // First true element.
      else {
        // Update double-compare state machine.
        if (SecondTrueElement == Undefined)
          SecondTrueElement = i;
        else
          SecondTrueElement = Overdefined;

        // Update range state machine.
        if (TrueRangeEnd == (int)i-1)
          TrueRangeEnd = i;
        else
          TrueRangeEnd = Overdefined;
      }
    } else {
      // Update the FalseElement state machine.
      if (FirstFalseElement == Undefined)
        FirstFalseElement = FalseRangeEnd = i; // First false element.
      else {
        // Update double-compare state machine.
        if (SecondFalseElement == Undefined)
          SecondFalseElement = i;
        else
          SecondFalseElement = Overdefined;

        // Update range state machine.
        if (FalseRangeEnd == (int)i-1)
          FalseRangeEnd = i;
        else
          FalseRangeEnd = Overdefined;
      }
    }

    // If this element is in range, update our magic bitvector.
    if (i < 64 && IsTrueForElt)
      MagicBitvector |= 1ULL << i;

    // If all of our states become overdefined, bail out early.  Since the
    // predicate is expensive, only check it every 8 elements.  This is only
    // really useful for really huge arrays.
    if ((i & 8) == 0 && i >= 64 && SecondTrueElement == Overdefined &&
        SecondFalseElement == Overdefined && TrueRangeEnd == Overdefined &&
        FalseRangeEnd == Overdefined)
      return nullptr;
  }

  // If a magic bitvector captures the entire comparison state
  // of this load, replace it with computation that does:
  //   ((magic_cst >> i) & 1) != 0
  {
    Type *Ty = nullptr;    // Look for an appropriate type:
    // - The type of Idx if the magic fits
    // - The smallest fitting legal type
    if (ArrayElementCount <= Idx->getType()->getIntegerBitWidth())
      Ty = Idx->getType();
    else
      Ty = DL.getSmallestLegalIntType(Init->getContext(), ArrayElementCount);

    if (Ty) {
      Idx = MaskIdx(Idx);
      Value *V = Builder.CreateIntCast(Idx, Ty, false);
      V = Builder.CreateLShr(ConstantInt::get(Ty, MagicBitvector), V);
      V = Builder.CreateAnd(ConstantInt::get(Ty, 1), V);
      return new ICmpInst(ICmpInst::ICMP_NE, V, ConstantInt::get(Ty, 0));
    }
  }

  return nullptr;
}

