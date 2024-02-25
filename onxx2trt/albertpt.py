from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Define a model wrapper
class ModelWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs.logits

# Initialize model and tokenizer
model_name = "JanSt/albert-base-v2_mbti-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input
text = "This is a sample text for ONNX conversion."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelWrapper(model_name).to(device)

# Move inputs to the same device as model
inputs = {k: v.to(device) for k, v in inputs.items()}

# Trace the model
scripted_model = torch.jit.trace(model, (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']))

# Save the scripted model
scripted_model.save("model.pt")

# Load the scripted model
scripted_model = torch.jit.load("model.pt")

# Assuming the model is still on the device from before, but if you restarted your kernel or changed devices, ensure to move it again
scripted_model.to(device)

# Prepare inputs for inference (usually you'd prepare new inputs, but for demonstration, we reuse the existing ones)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run inference
while True:
    with torch.no_grad():
        test_outputs = scripted_model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

# Print the result
print(test_outputs)