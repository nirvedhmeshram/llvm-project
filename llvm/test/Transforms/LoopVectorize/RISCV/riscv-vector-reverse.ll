; This is the loop in c++ being vectorize in this file with
;vector.reverse
;  #pragma clang loop vectorize_width(4, scalable)
;  for (int i = N-1; i >= 0; --i)
;    a[i] = b[i] + 1.0;

; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize,dce,instcombine -mtriple riscv64-linux-gnu \
; RUN:   -mattr=+v -debug-only=loop-vectorize -scalable-vectorization=on \
; RUN:   -riscv-v-vector-bits-min=128 -disable-output < %s 2>&1 | FileCheck %s

define void @vector_reverse_i64(ptr nocapture noundef writeonly %A, ptr nocapture noundef readonly %B, i32 noundef signext %n) {
; CHECK-LABEL: 'vector_reverse_i64'
; CHECK-NEXT:  LV: Loop hints: force=enabled width=vscale x 4 interleave=0
; CHECK-NEXT:  LV: Found a loop: for.body
; CHECK-NEXT:  LV: Found an induction variable.
; CHECK-NEXT:  LV: Found an induction variable.
; CHECK-NEXT:  LV: Did not find one integer induction var.
; CHECK-NEXT:  LV: We can vectorize this loop (with a runtime bound check)!
; CHECK-NEXT:  LV: Loop does not require scalar epilogue
; CHECK-NEXT:  LV: Found trip count: 0
; CHECK-NEXT:  LV: Found maximum trip count: 4294967295
; CHECK-NEXT:  LV: Scalable vectorization is available
; CHECK-NEXT:  LV: The max safe fixed VF is: 67108864.
; CHECK-NEXT:  LV: The max safe scalable VF is: vscale x 4294967295.
; CHECK-NEXT:  LV: Found uniform instruction: %cmp = icmp ugt i64 %indvars.iv, 1
; CHECK-NEXT:  LV: Found uniform instruction: %arrayidx = getelementptr inbounds i32, ptr %B, i64 %idxprom
; CHECK-NEXT:  LV: Found uniform instruction: %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %idxprom
; CHECK-NEXT:  LV: Found uniform instruction: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Found uniform instruction: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Found uniform instruction: %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:  LV: Found uniform instruction: %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  LV: Found uniform instruction: %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK-NEXT:  LV: Found uniform instruction: %i.0 = add nsw i32 %i.0.in8, -1
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %i.0 = add nsw i32 %i.0.in8, -1
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %arrayidx = getelementptr inbounds i32, ptr %B, i64 %idxprom
; CHECK-NEXT:  LV: Found an estimated cost of 13 for VF vscale x 4 For instruction: %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:  LV: Found an estimated cost of 2 for VF vscale x 4 For instruction: %add9 = add i32 %1, 1
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %idxprom
; CHECK-NEXT:  LV: Found an estimated cost of 13 for VF vscale x 4 For instruction: store i32 %add9, ptr %arrayidx3, align 4
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %cmp = icmp ugt i64 %indvars.iv, 1
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit, !llvm.loop !0
; CHECK-NEXT:  LV: Using user VF vscale x 4.
; CHECK-NEXT:  LV: Loop does not require scalar epilogue
; CHECK-NEXT:  LV: Scalarizing: %i.0 = add nsw i32 %i.0.in8, -1
; CHECK-NEXT:  LV: Scalarizing: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Scalarizing: %arrayidx = getelementptr inbounds i32, ptr %B, i64 %idxprom
; CHECK-NEXT:  LV: Scalarizing: %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %idxprom
; CHECK-NEXT:  LV: Scalarizing: %cmp = icmp ugt i64 %indvars.iv, 1
; CHECK-NEXT:  LV: Scalarizing: %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  VPlan 'Initial VPlan for VF={vscale x 4},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT:  vp<[[TC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.body.preheader>:
; CHECK-NEXT:    IR %0 = zext i32 %n to i64
; CHECK-NEXT:    EMIT vp<[[TC]]> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:      vp<[[DEV_IV:%.+]]> = DERIVED-IV ir<%n> + vp<[[CAN_IV]]> * ir<-1>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[DEV_IV]]>, ir<-1>
; CHECK-NEXT:      CLONE ir<%i.0> = add nsw vp<[[STEPS]]>, ir<-1>
; CHECK-NEXT:      CLONE ir<%idxprom> = zext ir<%i.0>
; CHECK-NEXT:      CLONE ir<%arrayidx> = getelementptr inbounds ir<%B>, ir<%idxprom>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = reverse-vector-pointer inbounds ir<%arrayidx>, vp<[[VF]]>
; CHECK-NEXT:      WIDEN ir<%1> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:      WIDEN ir<%add9> = add ir<%1>, ir<1>
; CHECK-NEXT:      CLONE ir<%arrayidx3> = getelementptr inbounds ir<%A>, ir<%idxprom>
; CHECK-NEXT:      vp<[[VEC_PTR2:%.+]]> = reverse-vector-pointer inbounds ir<%arrayidx3>, vp<[[VF]]>
; CHECK-NEXT:      WIDEN store vp<[[VEC_PTR2]]>, ir<%add9>
; CHECK-NEXT:      EMIT vp<[[CAN_IV_NEXT:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq vp<[[TC]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<for.cond.cleanup.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph:
; CHECK-NEXT:  Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.body>:
; CHECK-NEXT:    IR   %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:    IR   %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK:         IR   %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %i.0 = add nsw i32 %i.0.in8, -1
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %arrayidx = getelementptr inbounds i32, ptr %B, i64 %idxprom
; CHECK-NEXT:  LV: Found an estimated cost of 13 for VF vscale x 4 For instruction: %1 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:  LV: Found an estimated cost of 2 for VF vscale x 4 For instruction: %add9 = add i32 %1, 1
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %idxprom
; CHECK-NEXT:  LV: Found an estimated cost of 13 for VF vscale x 4 For instruction: store i32 %add9, ptr %arrayidx3, align 4
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %cmp = icmp ugt i64 %indvars.iv, 1
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit, !llvm.loop !0
; CHECK-NEXT:  LV(REG): Calculating max register usage:
; CHECK-NEXT:  LV(REG): At #0 Interval # 0
; CHECK-NEXT:  LV(REG): At #1 Interval # 1
; CHECK-NEXT:  LV(REG): At #2 Interval # 2
; CHECK-NEXT:  LV(REG): At #3 Interval # 2
; CHECK-NEXT:  LV(REG): At #4 Interval # 2
; CHECK-NEXT:  LV(REG): At #5 Interval # 3
; CHECK-NEXT:  LV(REG): At #6 Interval # 3
; CHECK-NEXT:  LV(REG): At #7 Interval # 3
; CHECK-NEXT:  LV(REG): At #9 Interval # 1
; CHECK-NEXT:  LV(REG): At #10 Interval # 2
; CHECK-NEXT:  LV(REG): VF = vscale x 4
; CHECK-NEXT:  LV(REG): Found max usage: 2 item
; CHECK-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 3 registers
; CHECK-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 2 registers
; CHECK-NEXT:  LV(REG): Found invariant usage: 1 item
; CHECK-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 1 registers
; CHECK-NEXT:  LV: The target has 31 registers of RISCV::GPRRC register class
; CHECK-NEXT:  LV: The target has 32 registers of RISCV::VRRC register class
; CHECK-NEXT:  LV: Loop does not require scalar epilogue
; CHECK-NEXT:  LV: Loop cost is 32
; CHECK-NEXT:  LV: IC is 1
; CHECK-NEXT:  LV: VF is vscale x 4
; CHECK-NEXT:  LV: Not Interleaving.
; CHECK-NEXT:  LV: Interleaving is not beneficial.
; CHECK-NEXT:  LV: Found a vectorizable loop (vscale x 4) in <stdin>
; CHECK-NEXT:  LEV: Epilogue vectorization is not profitable for this loop
; CHECK-NEXT:  Executing best plan with VF=vscale x 4, UF=1
; CHECK-NEXT:  VPlan 'Final VPlan for VF={vscale x 4},UF={1}' {
; CHECK-NEXT:  Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT:  vp<[[TC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.body.preheader>:
; CHECK-NEXT:    IR %0 = zext i32 %n to i64
; CHECK-NEXT:    EMIT vp<[[TC]]> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:      vp<[[DEV_IV:%.+]]> = DERIVED-IV ir<%n> + vp<[[CAN_IV]]> * ir<-1>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[DEV_IV]]>, ir<-1>
; CHECK-NEXT:      CLONE ir<%i.0> = add nsw vp<[[STEPS]]>, ir<-1>
; CHECK-NEXT:      CLONE ir<%idxprom> = zext ir<%i.0>
; CHECK-NEXT:      CLONE ir<%arrayidx> = getelementptr inbounds ir<%B>, ir<%idxprom>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = reverse-vector-pointer inbounds ir<%arrayidx>, vp<[[VF]]>
; CHECK-NEXT:      WIDEN ir<%13> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:      WIDEN ir<%add9> = add ir<%13>, ir<1>
; CHECK-NEXT:      CLONE ir<%arrayidx3> = getelementptr inbounds ir<%A>, ir<%idxprom>
; CHECK-NEXT:      vp<[[VEC_PTR2:%.+]]> = reverse-vector-pointer inbounds ir<%arrayidx3>, vp<[[VF]]>
; CHECK-NEXT:      WIDEN store vp<[[VEC_PTR2]]>, ir<%add9>
; CHECK-NEXT:      EMIT vp<[[CAN_IV_NEXT:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq vp<[[TC]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<for.cond.cleanup.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph:
; CHECK-NEXT:  Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.body>:
; CHECK-NEXT:    IR   %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:    IR   %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK:         IR   %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
; CHECK:  LV: Loop does not require scalar epilogue
;
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
  %i.0 = add nsw i32 %i.0.in8, -1
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  %add9 = add i32 %1, 1
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %idxprom
  store i32 %add9, ptr %arrayidx3, align 4
  %cmp = icmp ugt i64 %indvars.iv, 1
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

define void @vector_reverse_f32(ptr nocapture noundef writeonly %A, ptr nocapture noundef readonly %B, i32 noundef signext %n) {
; CHECK-LABEL: 'vector_reverse_f32'
; CHECK-NEXT:  LV: Loop hints: force=enabled width=vscale x 4 interleave=0
; CHECK-NEXT:  LV: Found a loop: for.body
; CHECK-NEXT:  LV: Found an induction variable.
; CHECK-NEXT:  LV: Found an induction variable.
; CHECK-NEXT:  LV: Found FP op with unsafe algebra.
; CHECK-NEXT:  LV: Did not find one integer induction var.
; CHECK-NEXT:  LV: We can vectorize this loop (with a runtime bound check)!
; CHECK-NEXT:  LV: Loop does not require scalar epilogue
; CHECK-NEXT:  LV: Found trip count: 0
; CHECK-NEXT:  LV: Found maximum trip count: 4294967295
; CHECK-NEXT:  LV: Scalable vectorization is available
; CHECK-NEXT:  LV: The max safe fixed VF is: 67108864.
; CHECK-NEXT:  LV: The max safe scalable VF is: vscale x 4294967295.
; CHECK-NEXT:  LV: Found uniform instruction: %cmp = icmp ugt i64 %indvars.iv, 1
; CHECK-NEXT:  LV: Found uniform instruction: %arrayidx = getelementptr inbounds float, ptr %B, i64 %idxprom
; CHECK-NEXT:  LV: Found uniform instruction: %arrayidx3 = getelementptr inbounds float, ptr %A, i64 %idxprom
; CHECK-NEXT:  LV: Found uniform instruction: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Found uniform instruction: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Found uniform instruction: %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:  LV: Found uniform instruction: %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  LV: Found uniform instruction: %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK-NEXT:  LV: Found uniform instruction: %i.0 = add nsw i32 %i.0.in8, -1
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %i.0 = add nsw i32 %i.0.in8, -1
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %arrayidx = getelementptr inbounds float, ptr %B, i64 %idxprom
; CHECK-NEXT:  LV: Found an estimated cost of 13 for VF vscale x 4 For instruction: %1 = load float, ptr %arrayidx, align 4
; CHECK-NEXT:  LV: Found an estimated cost of 4 for VF vscale x 4 For instruction: %conv1 = fadd float %1, 1.000000e+00
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %arrayidx3 = getelementptr inbounds float, ptr %A, i64 %idxprom
; CHECK-NEXT:  LV: Found an estimated cost of 13 for VF vscale x 4 For instruction: store float %conv1, ptr %arrayidx3, align 4
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %cmp = icmp ugt i64 %indvars.iv, 1
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit, !llvm.loop !0
; CHECK-NEXT:  LV: Using user VF vscale x 4.
; CHECK-NEXT:  LV: Loop does not require scalar epilogue
; CHECK-NEXT:  LV: Scalarizing: %i.0 = add nsw i32 %i.0.in8, -1
; CHECK-NEXT:  LV: Scalarizing: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Scalarizing: %arrayidx = getelementptr inbounds float, ptr %B, i64 %idxprom
; CHECK-NEXT:  LV: Scalarizing: %arrayidx3 = getelementptr inbounds float, ptr %A, i64 %idxprom
; CHECK-NEXT:  LV: Scalarizing: %cmp = icmp ugt i64 %indvars.iv, 1
; CHECK-NEXT:  LV: Scalarizing: %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  VPlan 'Initial VPlan for VF={vscale x 4},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT:  vp<[[TC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.body.preheader>:
; CHECK-NEXT:    IR %0 = zext i32 %n to i64
; CHECK-NEXT:    EMIT vp<[[TC]]> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:      vp<[[DEV_IV:%.+]]> = DERIVED-IV ir<%n> + vp<[[CAN_IV]]> * ir<-1>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[DEV_IV]]>, ir<-1>
; CHECK-NEXT:      CLONE ir<%i.0> = add nsw vp<[[STEPS]]>, ir<-1>
; CHECK-NEXT:      CLONE ir<%idxprom> = zext ir<%i.0>
; CHECK-NEXT:      CLONE ir<%arrayidx> = getelementptr inbounds ir<%B>, ir<%idxprom>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = reverse-vector-pointer inbounds ir<%arrayidx>, vp<[[VF]]>
; CHECK-NEXT:      WIDEN ir<%1> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:      WIDEN ir<%conv1> = fadd ir<%1>, ir<1.000000e+00>
; CHECK-NEXT:      CLONE ir<%arrayidx3> = getelementptr inbounds ir<%A>, ir<%idxprom>
; CHECK-NEXT:      vp<[[VEC_PTR2:%.+]]> = reverse-vector-pointer inbounds ir<%arrayidx3>, vp<[[VF]]>
; CHECK-NEXT:      WIDEN store vp<[[VEC_PTR2]]>, ir<%conv1>
; CHECK-NEXT:      EMIT vp<[[CAN_IV_NEXT:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq vp<[[TC]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<for.cond.cleanup.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph:
; CHECK-NEXT:  Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.body>:
; CHECK-NEXT:    IR   %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:    IR   %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK:         IR   %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %i.0 = add nsw i32 %i.0.in8, -1
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %idxprom = zext i32 %i.0 to i64
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %arrayidx = getelementptr inbounds float, ptr %B, i64 %idxprom
; CHECK-NEXT:  LV: Found an estimated cost of 13 for VF vscale x 4 For instruction: %1 = load float, ptr %arrayidx, align 4
; CHECK-NEXT:  LV: Found an estimated cost of 4 for VF vscale x 4 For instruction: %conv1 = fadd float %1, 1.000000e+00
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: %arrayidx3 = getelementptr inbounds float, ptr %A, i64 %idxprom
; CHECK-NEXT:  LV: Found an estimated cost of 13 for VF vscale x 4 For instruction: store float %conv1, ptr %arrayidx3, align 4
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %cmp = icmp ugt i64 %indvars.iv, 1
; CHECK-NEXT:  LV: Found an estimated cost of 1 for VF vscale x 4 For instruction: %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  LV: Found an estimated cost of 0 for VF vscale x 4 For instruction: br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit, !llvm.loop !0
; CHECK-NEXT:  LV(REG): Calculating max register usage:
; CHECK-NEXT:  LV(REG): At #0 Interval # 0
; CHECK-NEXT:  LV(REG): At #1 Interval # 1
; CHECK-NEXT:  LV(REG): At #2 Interval # 2
; CHECK-NEXT:  LV(REG): At #3 Interval # 2
; CHECK-NEXT:  LV(REG): At #4 Interval # 2
; CHECK-NEXT:  LV(REG): At #5 Interval # 3
; CHECK-NEXT:  LV(REG): At #6 Interval # 3
; CHECK-NEXT:  LV(REG): At #7 Interval # 3
; CHECK-NEXT:  LV(REG): At #9 Interval # 1
; CHECK-NEXT:  LV(REG): At #10 Interval # 2
; CHECK-NEXT:  LV(REG): VF = vscale x 4
; CHECK-NEXT:  LV(REG): Found max usage: 2 item
; CHECK-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 3 registers
; CHECK-NEXT:  LV(REG): RegisterClass: RISCV::VRRC, 2 registers
; CHECK-NEXT:  LV(REG): Found invariant usage: 1 item
; CHECK-NEXT:  LV(REG): RegisterClass: RISCV::GPRRC, 1 registers
; CHECK-NEXT:  LV: The target has 31 registers of RISCV::GPRRC register class
; CHECK-NEXT:  LV: The target has 32 registers of RISCV::VRRC register class
; CHECK-NEXT:  LV: Loop does not require scalar epilogue
; CHECK-NEXT:  LV: Loop cost is 34
; CHECK-NEXT:  LV: IC is 1
; CHECK-NEXT:  LV: VF is vscale x 4
; CHECK-NEXT:  LV: Not Interleaving.
; CHECK-NEXT:  LV: Interleaving is not beneficial.
; CHECK-NEXT:  LV: Found a vectorizable loop (vscale x 4) in <stdin>
; CHECK-NEXT:  LEV: Epilogue vectorization is not profitable for this loop
; CHECK-NEXT:  Executing best plan with VF=vscale x 4, UF=1
; CHECK-NEXT:  VPlan 'Final VPlan for VF={vscale x 4},UF={1}' {
; CHECK-NEXT:  Live-in vp<[[VF:%.+]]> = VF
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VEC_TC:%.+]]> = vector-trip-count
; CHECK-NEXT:  vp<[[TC:%.+]]> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.body.preheader>:
; CHECK-NEXT:    IR %0 = zext i32 %n to i64
; CHECK-NEXT:    EMIT vp<[[TC]]> = EXPAND SCEV (zext i32 %n to i64)
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:      vp<[[DEV_IV:%.+]]> = DERIVED-IV ir<%n> + vp<[[CAN_IV]]> * ir<-1>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[DEV_IV]]>, ir<-1>
; CHECK-NEXT:      CLONE ir<%i.0> = add nsw vp<[[STEPS]]>, ir<-1>
; CHECK-NEXT:      CLONE ir<%idxprom> = zext ir<%i.0>
; CHECK-NEXT:      CLONE ir<%arrayidx> = getelementptr inbounds ir<%B>, ir<%idxprom>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = reverse-vector-pointer inbounds ir<%arrayidx>, vp<[[VF]]>
; CHECK-NEXT:      WIDEN ir<%13> = load vp<[[VEC_PTR]]>
; CHECK-NEXT:      WIDEN ir<%conv1> = fadd ir<%13>, ir<1.000000e+00>
; CHECK-NEXT:      CLONE ir<%arrayidx3> = getelementptr inbounds ir<%A>, ir<%idxprom>
; CHECK-NEXT:      vp<[[VEC_PTR:%.+]]> = reverse-vector-pointer inbounds ir<%arrayidx3>, vp<[[VF]]>
; CHECK-NEXT:      WIDEN store vp<[[VEC_PTR]]>, ir<%conv1>
; CHECK-NEXT:      EMIT vp<[[CAN_IV_NEXT:%.+]]> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<[[CAN_IV_NEXT]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
; CHECK-NEXT:  Successor(s): middle.block
; CHECK-EMPTY:
; CHECK-NEXT:  middle.block:
; CHECK-NEXT:    EMIT vp<[[CMP:%.+]]> = icmp eq vp<[[TC]]>, vp<[[VEC_TC]]>
; CHECK-NEXT:    EMIT branch-on-cond vp<[[CMP]]>
; CHECK-NEXT:  Successor(s): ir-bb<for.cond.cleanup.loopexit>, scalar.ph
; CHECK-EMPTY:
; CHECK-NEXT:  scalar.ph:
; CHECK-NEXT:  Successor(s): ir-bb<for.body>
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.body>:
; CHECK-NEXT:    IR   %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT:    IR   %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
; CHECK:         IR   %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK-NEXT:  No successors
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<for.cond.cleanup.loopexit>:
; CHECK-NEXT:  No successors
; CHECK-NEXT:  }
; CHECK:  LV: Loop does not require scalar epilogue
;
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %0 = zext i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %i.0.in8 = phi i32 [ %n, %for.body.preheader ], [ %i.0, %for.body ]
  %i.0 = add nsw i32 %i.0.in8, -1
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds float, ptr %B, i64 %idxprom
  %1 = load float, ptr %arrayidx, align 4
  %conv1 = fadd float %1, 1.000000e+00
  %arrayidx3 = getelementptr inbounds float, ptr %A, i64 %idxprom
  store float %conv1, ptr %arrayidx3, align 4
  %cmp = icmp ugt i64 %indvars.iv, 1
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
