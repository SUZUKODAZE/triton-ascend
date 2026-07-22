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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_SINK_I1_PRODUCERS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_SINK_I1_PRODUCERS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton {

// SinkI1ProducersPass
// =========================================================================
// For every op that produces an i1-typed result, rewrite each of its i1 uses
// by cloning the producer right before the consuming op. The original
// producer is then erased.
//
// Effect: every (i1 producer, i1 consumer) pair becomes a back-to-back,
// one-to-one adjacency, no matter where the original producer sat.
//
// Safety: SSA dominance is guaranteed by construction -- every OpOperand
// use of the producer exists iff the producer dominates the consumer.
// Producers are restricted to side-effect-free ops (no regions, no memory
// effects) so duplication is safe.
class SinkI1ProducersIntoUsersPass : public PassWrapper<SinkI1ProducersIntoUsersPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SinkI1ProducersIntoUsersPass);

    SinkI1ProducersIntoUsersPass() = default;
    void runOnOperation() override;

    llvm::StringRef getArgument() const final {
        return "sink-i1-producers-into-users";
    }
    llvm::StringRef getDescription() const final {
        return "Sink i1-producing ops next to each of their i1 uses";
    }
};

std::unique_ptr<OperationPass<ModuleOp>> createSinkI1ProducersIntoUsersPass();

void registerSinkI1ProducersIntoUsersPasses();

} // namespace mlir::triton

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_SINK_I1_PRODUCERS_H