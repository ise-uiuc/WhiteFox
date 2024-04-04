PreservedAnalyses GlobalDCEPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool Changed = false;

  NumFunctions += DeadFunctions.size();
  for (Function *F : DeadFunctions) {
    if (!F->use_empty()) {
      // Virtual functions might still be referenced by one or more vtables,
      // but if we've proven them to be unused then it's safe to replace the
      // virtual function pointers with null, allowing us to remove the
      // function itself.
      ++NumVFuncs;      // Detect vfuncs that are referenced as "relative pointers" which are used
      // in Swift vtables, i.e. entries in the form of:
      //
      //   i32 trunc (i64 sub (i64 ptrtoint @f, i64 ptrtoint ...)) to i32)
      //
      // In this case, replace the whole "sub" expression with constant 0 to
      // avoid leaving a weird sub(0, symbol) expression behind.
      replaceRelativePointerUsersWithZero(F);

      F->replaceNonMetadataUsesWith(ConstantPointerNull::get(F->getType()));
      LLVM_DEBUG(dbgs() << "F->replaceNonMetadataUsesWith(ConstantPointerNull::get(F->getType()));\n");
    }
    EraseUnusedGlobalValue(F);
  }

  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

