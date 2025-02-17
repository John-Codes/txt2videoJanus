# test_imports.py
from JRX import JRX
from JRX.janus.models import MultiModalityCausalLM

model = JRX()
print(model.inference("Test"))
# test and show me the image generated
