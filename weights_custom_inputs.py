import json
import random
import math
import os

def calculate_ideal_base(inputs: list, target: float) -> float:

    weighted_sum = sum(val * (i + 1) for i, val in enumerate(inputs))
    base = (weighted_sum * 0.25) + (target * 1.5) + math.sin(target * math.pi)
    
    return base

def generate_json_dataset(num_inputs: int, num_samples: int, output_dir: str = "."):
    
    filename = os.path.join(output_dir, f"weights_{num_inputs}inputs.json")
    samples = []
    
    for _ in range(num_samples):
        inputs = [random.uniform(-2.0, 2.0) for _ in range(num_inputs)]
        
        target = random.uniform(0.0, 1.0)
        
        ideal_base = calculate_ideal_base(inputs, target)
        
        samples.append({
            "inputs": inputs,
            "target": target,
            "ideal_base": round(ideal_base, 6)
        })
        
    dataset = {
        "num_inputs": num_inputs,
        "samples": samples
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
        
    print("\n" + "━"*40)
    print(f"✅ Generated {num_samples} samples for a {num_inputs}-input network.")
    print(f"📄 Saved successfully to: {filename}")
    print("━"*40 + "\n")

if __name__ == "__main__":
    print("🧠 Hypernetwork Training Data Generator 🧠")
    print("This script generates JSON files compatible with your PatternPredictor.")
    
    try:
        inputs_str = input("Enter the number of inputs (e.g., 5): ").strip()
        num_inputs = int(inputs_str)
        
        if num_inputs < 1:
            raise ValueError("Number of inputs must be at least 1.")
            
        samples_str = input("Enter the number of samples to generate (e.g., 200): ").strip()
        num_samples = int(samples_str)
        
        if num_samples < 1:
            raise ValueError("Number of samples must be at least 1.")
            
        generate_json_dataset(num_inputs, num_samples)
        
    except ValueError as e:
        print(f"\n❌ Invalid input: {e}. Please enter positive whole numbers.")
    except KeyboardInterrupt:
        print("\n\n🛑 Generation cancelled by user.")