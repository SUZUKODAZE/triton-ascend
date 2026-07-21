/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/SinkI1Producers.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// Returns true if `op` produces an i1-typed result.
bool isI1Producer(Operation *op)
{
    return llvm::any_of(op->getResults(), [](Value v) {
        if (auto intTy = v.getType().dyn_cast<IntegerType>())
            return intTy.getWidth() == 1;
        return false;
    });
}

// Returns true if `op` is safe to duplicate: no regions, no memory effects,
// no recursive side-effect trait. arith.cmpi/cmpf/andi/ori/xori/trunci and
// similar pure value ops fall in this set; scf.if, func.func, loads/stores,
// and any op with regions or memory effects are excluded.
bool isPureAndRegionless(Operation *op)
{
    if (op->getNumRegions() != 0)
        return false;
    if (op->hasTrait<OpTrait::HasRecursiveSideEffects>())
        return false;
    if (isa<MemoryEffectOpInterface>(op))
        return false;
    return true;
}

} // namespace

namespace mlir::triton {

void SinkI1ProducersPass::runOnOperation()
{
    ModuleOp moduleOp = getOperation();

    // Collect all candidate producers first so we can safely mutate the IR
    // (erase) without invalidating the walk iterator.
    SmallVector<Operation *> producers;
    moduleOp.walk([&](Operation *op) {
        if (isI1Producer(op) && isPureAndRegionless(op))
            producers.push_back(op);
    });

    for (Operation *p : producers) {
        // Gather every i1-typed use of p.
        SmallVector<std::pair<OpOperand *, unsigned>> i1Uses;
        for (OpOperand &use : p->getUses()) {
            auto intTy = use.get().getType().dyn_cast<IntegerType>();
            if (intTy && intTy.getWidth() == 1) {
                unsigned idx = use.getOperandNumber();
                i1Uses.push_back({&use, idx});
            }
        }

        if (i1Uses.empty()) {
            p->erase();
            continue;
        }

        // For every i1 use, clone p right before the consuming op and
        // redirect the use to the clone. SSA dominance is guaranteed because
        // the use itself only exists if p dominates the consumer.
        for (auto [use, resultIdx] : i1Uses) {
            Operation *consumer = use->getOwner();
            Operation *cloned = p->clone();
            cloned->moveBefore(consumer);
            use->set(cloned->getResult(resultIdx));
        }

        // Original producer is no longer referenced by any i1 use.
        // If it has no remaining non-i1 users either, drop it.
        if (p->use_empty())
            p->erase();
    }
}

std::unique_ptr<OperationPass<ModuleOp>> createSinkI1ProducersPass()
{
    return std::make_unique<SinkI1ProducersPass>();
}

void registerSinkI1ProducersPasses()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return createSinkI1ProducersPass();
    });
}

} // namespace mlir::triton