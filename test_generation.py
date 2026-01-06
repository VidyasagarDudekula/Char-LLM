
import torch
from gpt import GPTCharecterModel, device, decode

def run_test():
    print("Initializing model...")
    model = GPTCharecterModel()
    model = model.to(device)
    
    print("Loading weights from char-model.pth...")
    try:
        model.load_state_dict(torch.load('char-model.pth', map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: char-model.pth not found. Please run gpt.py first to train the model.")
        return

    model.eval() # Set to evaluation mode

    # Start with zero token id, batch size 1
    print("Starting generation with zero token id...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate 10000 characters
    max_length = 10000
    print(f"Generating {max_length} characters...")
    
    with torch.no_grad():
        generated_ids = model.generate(context, max_length=max_length)[0].tolist()
    
    generated_text = decode(generated_ids)
    
    output_file = "test_output_10000.txt"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(generated_text)
        
    print(f"Generation complete. Output saved to {output_file}")
    print("First 200 characters of generated text:")
    print("-" * 50)
    print(generated_text[:200])
    print("-" * 50)

if __name__ == "__main__":
    run_test()
