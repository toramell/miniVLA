# miniVLA

Vision-Language-Action (VLA) minimal implementation for:
- UCF101 pretraining
- LIBERO robot policy finetuning
- Multi-GPU (A100) support

---

## 1. Environment Setup

```bash
conda create -n miniVLA python=3.10
conda activate miniVLA
pip install -r requirements.txt

2. UCF101 Pretraining
python train_ssv2.py

3. LIBERO Fine-tuning
python train_libero.py --multi_gpu --auto_select_a100

4. Inference
python libero_inference.py --checkpoint model.pt

5. Multi-GPU

Automatic A100 detection:

--multi_gpu --auto_select_a100


