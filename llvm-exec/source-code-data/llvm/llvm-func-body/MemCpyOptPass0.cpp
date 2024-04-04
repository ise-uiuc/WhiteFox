bool MemCpyOptPass::iterateOnFunction(Function &F) {
  bool MadeChange = false;

  // Walk all instruction in the function.
  for (BasicBlock &BB : F) {
    // Skip unreachable blocks. For example processStore assumes that an
    // instruction in a BB can't be dominated by a later instruction in the
    // same BB (which is a scenario that can happen for an unreachable BB that
    // has itself as a predecessor).
    if (!DT->isReachableFromEntry(&BB))
      continue;    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end(); BI != BE;) {
        // Avoid invalidating the iterator.
      Instruction *I = &*BI++;

      bool RepeatInstruction = false;

      if (auto *SI = dyn_cast<StoreInst>(I))
        MadeChange |= processStore(SI, BI);
      else if (auto *M = dyn_cast<MemSetInst>(I))
        RepeatInstruction = processMemSet(M, BI);
      else if (auto *M = dyn_cast<MemCpyInst>(I))
        RepeatInstruction = processMemCpy(M, BI);
      else if (auto *M = dyn_cast<MemMoveInst>(I))
        RepeatInstruction = processMemMove(M);
      else if (auto *CB = dyn_cast<CallBase>(I)) {
        for (unsigned i = 0, e = CB->arg_size(); i != e; ++i)
          if (CB->isByValArgument(i))
            MadeChange |= processByValArgument(*CB, i);
      }

      // Reprocess the instruction if desired.
      if (RepeatInstruction) {
        if (BI != BB.begin())
          --BI;
        MadeChange = true;
      }
    }
  }

  return MadeChange;
}

bool MemCpyOptPass::processStore(StoreInst *SI, BasicBlock::iterator &BBI) {

  if (!SI->isSimple()) return false;

  // Avoid merging nontemporal stores since the resulting
  // memcpy/memset would not be able to preserve the nontemporal hint.
  // In theory we could teach how to propagate the !nontemporal metadata to
  // memset calls. However, that change would force the backend to
  // conservatively expand !nontemporal memset calls back to sequences of
  // store instructions (effectively undoing the merging).
  if (SI->getMetadata(LLVMContext::MD_nontemporal))
    return false;

  // Not all the transforms below are correct for non-integral pointers, bail
  // until we've audited the individual pieces.
  if (DL.isNonIntegralPointerType(StoredVal->getType()->getScalarType()))
    return false;

  // The following code creates memset intrinsics out of thin air. Don't do
  // this if the corresponding libfunc is not available.
  // TODO: We should really distinguish between libcall availability and
  // our ability to introduce intrinsics.
  if (!(TLI->has(LibFunc_memset) || EnableMemCpyOptWithoutLibcalls))
    return false;

  // Ensure that the value being stored is something that can be memset'able a
  // byte at a time like "0" or "-1" or any width, as well as things like
  // 0xA0A0A0A0 and 0.0.
  auto *V = SI->getOperand(0);
  if (Value *ByteVal = isBytewiseValue(V, DL)) {
    if (Instruction *I = tryMergingIntoMemset(SI, SI->getPointerOperand(),
                                              ByteVal)) {
      BBI = I->getIterator(); // Don't invalidate iterator.
      return true;
    }    // If we have an aggregate, we try to promote it to memset regardless
    // of opportunity for merging as it can expose optimization opportunities
    // in subsequent passes.
    auto *T = V->getType();
    if (T->isAggregateType()) {
      uint64_t Size = DL.getTypeStoreSize(T);
      IRBuilder<> Builder(SI);
      auto *M = Builder.CreateMemSet(SI->getPointerOperand(), ByteVal, Size,
                                     SI->getAlign());
      M->copyMetadata(*SI, LLVMContext::MD_DIAssignID);

      LLVM_DEBUG(dbgs() << "Promoting " << *SI << " to " << *M << "\n");

      // The newly inserted memset is immediately overwritten by the original
      // store, so we do not need to rename uses.
      auto *StoreDef = cast<MemoryDef>(MSSA->getMemoryAccess(SI));
      auto *NewAccess = MSSAU->createMemoryAccessBefore(
          M, StoreDef->getDefiningAccess(), StoreDef);
      MSSAU->insertDef(cast<MemoryDef>(NewAccess), /*RenameUses=*/false);

      eraseInstruction(SI);
      NumMemSetInfer++;

      // Make sure we do not invalidate the iterator.
      BBI = M->getIterator();
      return true;
    }
  }

  return false;
}

