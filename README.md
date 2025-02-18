# Use Janus 7B Pro in one line of code!

Effortlessly integrate Janus 7B Pro, a state-of-the-art multimodal AI model, with just one line of code. This powerful model excels in both understanding and generating multimodal content, including text and images. By decoupling visual encoding, Janus 7B Pro offers unmatched flexibility and performance, surpassing competitors like DALL-E 3 and Stable Diffusion in various benchmarks123.

## Installation

```bash
pip install jrx-ai==0.1.0
```

## Usage

```python
# test_imports.py
from JRX import JRX
from JRX.janus.models import MultiModalityCausalLM

model = JRX()
print(model.inference("Test"))
```

Experience the future of AI with Janus 7B Pro today!

## GPU Requirements

The following table outlines the recommended GPU memory requirements for different Janus model sizes:

| Model Size | Minimum GPU Memory | Recommended GPU Memory | Notes |
|------------|-------------------|------------------------|-------|
| Janus 1B   | 8 GB             | 16 GB                  | Can run with batch size 1 on lower-end GPUs |
| Janus 1.3B | 12 GB            | 24 GB                  | Requires more VRAM than 1B version          |
| Janus 7B   | 16 GB            | 32 GB+                 | Needs substantial VRAM for optimal performance |

### Important Considerations

These recommendations are based on typical transformer architecture requirements. Actual memory usage depends on:
- Batch size
- Sequence length
- Specific implementation details
- Other running processes

For production environments, consider adding 20-30% overhead to recommended amounts. Always test with your specific use case, as actual requirements may vary.

### Note
Runs well on Nvidia RTX 4090
These recommendations are based on industry standards for transformer models of similar sizes. For precise requirements, consult the official documentation or contact the model developers directly.
