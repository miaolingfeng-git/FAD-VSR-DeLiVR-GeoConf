# FAD-VSR-DeLiVR-GeoConf

这是 FAD-VSR-DeLiVR 的第一种重构方案：将 Lie/几何思想从对齐后的全局时间注意力移动到 `FrequencyGuidedAlign` 的置信度分支。

## 改进点

1. 使用中心特征、已对齐邻帧特征和高频特征预测相对 SE(2) 参数：旋转角、x 平移、y 平移；
2. 将预测 SE(2) 位移与 `flow + deformable residual offset` 的局部位移进行比较；
3. 由几何误差生成 geometry consistency score；
4. 用该 score 修正 alignment confidence logits；
5. 用小权重 `geometry_consistency=0.01` 约束预测几何与 flow（flow target stop-gradient）；
6. 几何 confidence 的强度从小值开始，尽量保持训练初始状态接近原始 FAD-VSR。

## 实验协议

- 训练、验证、测试序列与之前的 50 epoch 实验一致；
- 7 帧输入，×4 超分；
- batch size 2，50 epoch，AdamW，学习率 2e-4；
- 测试使用同一套输出和统一指标脚本。
