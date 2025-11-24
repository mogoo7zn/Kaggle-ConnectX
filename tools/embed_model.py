"""
Convert PyTorch model to embedded Base64 format
用于将 best_model.pth 转换为可嵌入 main.py 的格式
"""
import torch
import base64
import io
import os

def embed_model(model_path='best_model.pth', output_path='model_weights_embedded.txt'):
    """
    Convert model to Base64 encoded string
    
    Args:
        model_path: Path to the .pth model file
        output_path: Path to save the base64 encoded text
    """
    # Load the model
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # Convert to bytes
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    model_bytes = buffer.read()

    # Encode to base64
    model_base64 = base64.b64encode(model_bytes).decode('utf-8')

    # Print statistics
    print(f"\nModel Statistics:")
    print(f"  Original size: {len(model_bytes):,} bytes")
    print(f"  Base64 size: {len(model_base64):,} characters")
    print(f"  Size increase: {(len(model_base64)/len(model_bytes)-1)*100:.1f}%")
    print(f"\nFirst 100 chars: {model_base64[:100]}")

    # Save to file
    with open(output_path, 'w') as f:
        f.write(model_base64)

    print(f"\nSaved to: {output_path}")
    return model_base64

if __name__ == '__main__':
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else '../submission/best_model.pth'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'model_weights_embedded.txt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    embed_model(model_path, output_path)

