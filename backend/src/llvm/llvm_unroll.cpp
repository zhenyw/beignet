/*
 * Copyright © 2012 Intel Corporation
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 */

#include "llvm/Config/llvm-config.h"
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 5
#include <set>
#if LLVM_VERSION_MINOR <= 2
#include "llvm/Function.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#else
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#endif  /* LLVM_VERSION_MINOR <= 2 */
#include "llvm/Pass.h"
#if LLVM_VERSION_MINOR <= 1
#include "llvm/Support/IRBuilder.h"
#elif LLVM_VERSION_MINOR == 2
#include "llvm/IRBuilder.h"
#else
#include "llvm/IR/IRBuilder.h"
#endif /* LLVM_VERSION_MINOR <= 1 */
#include "llvm/Support/raw_ostream.h"
#include "llvm/PassManager.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/llvm_gen_backend.hpp"
#include "sys/map.hpp"


using namespace llvm;

namespace gbe {
    class CustomLoopUnroll : public LoopPass
    {
    public:
      static char ID;
      CustomLoopUnroll() :
       LoopPass(ID) {}

      void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.addRequired<LoopInfo>();
        AU.addPreserved<LoopInfo>();
        AU.addRequiredID(LoopSimplifyID);
        AU.addPreservedID(LoopSimplifyID);
        AU.addRequiredID(LCSSAID);
        AU.addPreservedID(LCSSAID);
        AU.addRequired<ScalarEvolution>();
        AU.addPreserved<ScalarEvolution>();
      // FIXME: Loop unroll requires LCSSA. And LCSSA requires dom info.
      // If loop unroll does not preserve dom info then LCSSA pass on next
      // loop will receive invalid dom info.
      // For now, recreate dom info, if loop is unrolled.
      AU.addPreserved<DominatorTreeWrapperPass>();

      }

      // Returns the value associated with the given metadata node name (for
      // example, "llvm.loop.unroll.count").  If no such named metadata node
      // exists, then nullptr is returned.
      static const ConstantInt *GetUnrollMetadataValue(const Loop *L,
                                                     StringRef Name) {
        MDNode *LoopID = L->getLoopID();
        if (!LoopID) return nullptr;
        // First operand should refer to the loop id itself.
        assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
        assert(LoopID->getOperand(0) == LoopID && "invalid loop id");
        for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
          const MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
          if (!MD) continue;
          const MDString *S = dyn_cast<MDString>(MD->getOperand(0));
          if (!S) continue;
          if (Name.equals(S->getString())) {
            assert(MD->getNumOperands() == 2 &&
                   "Unroll hint metadata should have two operands.");
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 6
            return mdconst::extract<ConstantInt>(MD->getOperand(1));
#else
            return cast<ConstantInt>(MD->getOperand(1));
#endif
          }
        }
        return nullptr;
      }

      void setUnrollID(Loop *L, bool enable) {
        if (!enable && disabledLoops.find(L) != disabledLoops.end())
           return;
        LLVMContext &Context = L->getHeader()->getContext();
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 6
        SmallVector<Metadata *, 2> forceUnroll;
        forceUnroll.push_back(MDString::get(Context, "llvm.loop.unroll.enable"));
        forceUnroll.push_back(ConstantAsMetadata::get(ConstantInt::get(Type::getInt1Ty(Context), enable)));
        MDNode *forceUnrollNode = MDNode::get(Context, forceUnroll);
        SmallVector<Metadata *, 4> Vals;
        Vals.push_back(NULL);
        Vals.push_back(forceUnrollNode);
#else
        SmallVector<Value *, 2> forceUnroll;
        forceUnroll.push_back(MDString::get(Context, "llvm.loop.unroll.enable"));
        forceUnroll.push_back(ConstantInt::get(Type::getInt1Ty(Context), enable));
        MDNode *forceUnrollNode = MDNode::get(Context, forceUnroll);
        SmallVector<Value *, 4> Vals;
        Vals.push_back(NULL);
        Vals.push_back(forceUnrollNode);
#endif
        MDNode *NewLoopID = MDNode::get(Context, Vals);
        // Set operand 0 to refer to the loop id itself.
        NewLoopID->replaceOperandWith(0, NewLoopID);
        L->setLoopID(NewLoopID);
        if (!enable)
          disabledLoops.insert(L);
      }

      static bool hasPrivateLoadStore(Loop *L) {
        const std::vector<Loop*> subLoops = L->getSubLoops();
        std::set<BasicBlock*> subBlocks, blocks;

        for(auto l : subLoops)
          for(auto bb : l->getBlocks())
            subBlocks.insert(bb);
        for(auto bb : L->getBlocks())
          if (subBlocks.find(bb) == subBlocks.end())
            blocks.insert(bb);
        for(auto bb : blocks) {
          for (BasicBlock::iterator inst = bb->begin(), instE = bb->end(); inst != instE; ++inst) {
            unsigned addrSpace = -1;
            if (isa<LoadInst>(*inst)) {
              LoadInst *ld = cast<LoadInst>(&*inst);
              addrSpace = ld->getPointerAddressSpace();
            }
            else if (isa<StoreInst>(*inst)) {
              StoreInst *st = cast<StoreInst>(&*inst);
              addrSpace = st->getPointerAddressSpace();
            }
            if (addrSpace == 0)
              return true;
          }
        }
        return false;
      }
      // If one loop has very large self trip count
      // we don't want to unroll it.
      // self trip count means trip count divide by the parent's trip count. for example
      // for (int i = 0; i < 16; i++) {
      //   for (int j = 0; j < 4; j++) {
      //     for (int k = 0; k < 2; k++) {
      //       ...
      //     }
      //     ...
      //   }
      // The inner loops j and k could be unrolled, but the loop i will not be unrolled.
      // The return value true means the L could be unrolled, otherwise, it could not
      // be unrolled.
      bool handleParentLoops(Loop *L, LPPassManager &LPM) {
        Loop *currL = L;
        ScalarEvolution *SE = &getAnalysis<ScalarEvolution>();
        BasicBlock *ExitBlock = currL->getLoopLatch();
        if (!ExitBlock || !L->isLoopExiting(ExitBlock))
          ExitBlock = currL->getExitingBlock();

        unsigned currTripCount = 0;
        bool shouldUnroll = true;
        if (ExitBlock)
          currTripCount = SE->getSmallConstantTripCount(L, ExitBlock);

        while(currL) {
          Loop *parentL = currL->getParentLoop();
          unsigned parentTripCount = 0;
          if (parentL) {
            BasicBlock *parentExitBlock = parentL->getLoopLatch();
            if (!parentExitBlock || !parentL->isLoopExiting(parentExitBlock))
              parentExitBlock = parentL->getExitingBlock();

            if (parentExitBlock)
              parentTripCount = SE->getSmallConstantTripCount(parentL, parentExitBlock);
          }
          if ((parentTripCount != 0 && currTripCount / parentTripCount > 16) ||
              (currTripCount > 32)) {
            if (currL == L)
              shouldUnroll = false;
            setUnrollID(currL, false);
            if (currL != L)
              LPM.deleteLoopFromQueue(currL);
          }
          currL = parentL;
          currTripCount = parentTripCount;
        }
        return shouldUnroll;
      }

      // Analyze the outermost BBs of this loop, if there are
      // some private load or store, we change it's loop meta data
      // to indicate more aggresive unrolling on it.
      virtual bool runOnLoop(Loop *L, LPPassManager &LPM) {
        const ConstantInt *Enable = GetUnrollMetadataValue(L, "llvm.loop.unroll.enable");
        if (Enable)
          return false;
        const ConstantInt *Count = GetUnrollMetadataValue(L, "llvm.loop.unroll.count");
        if (Count)
          return false;

        if (!handleParentLoops(L, LPM))
          return false;

        if (!hasPrivateLoadStore(L))
          return false;
        setUnrollID(L, true);
        return true;
      }

      virtual const char *getPassName() const {
        return "SPIR backend: custom loop unrolling pass";
      }
    private:
      std::set<Loop *> disabledLoops;

    };

    char CustomLoopUnroll::ID = 0;

    LoopPass *createCustomLoopUnrollPass() {
      return new CustomLoopUnroll();
    }
} // end namespace
#endif
