1. basic_PTQ_demo.py是一个PT2E PTQ示例，可以对着分析一下算子图

2. PTQ_XNNPACK.py是使用XNNPACK Quantizer进行PTQ量化的示例

3. PTQ_Self_Quantizer.py是使用自定义的量化器进行PTQ量化的示例，目标是实现XNNPACK Quantizer一样的效果（可能还有部分细节没有完全对齐？但是效果还行）

4. QAT.py是一个PT2E QAT示例