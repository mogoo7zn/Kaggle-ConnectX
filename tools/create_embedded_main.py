"""
Create main.py with embedded model weights
用于生成包含嵌入模型权重的 main.py 文件
"""
import os
import sys

def create_embedded_main(
    original_main='../submission/main_backup.py',
    model_base64_file='model_weights_embedded.txt',
    output_file='../submission/main.py'
):
    """
    Create main.py with embedded model weights
    
    Args:
        original_main: Path to original main.py (without embedded weights)
        model_base64_file: Path to base64 encoded model file
        output_file: Path to save the new main.py
    """
    print("Creating embedded main.py...")
    
    # Read the original main.py
    print(f"Reading: {original_main}")
    with open(original_main, 'r', encoding='utf-8') as f:
        main_content = f.read()

    # Read the base64 encoded model
    print(f"Reading: {model_base64_file}")
    with open(model_base64_file, 'r') as f:
        model_base64 = f.read()

    # Split the base64 string into chunks of 80 characters for readability
    chunk_size = 80
    chunks = [model_base64[i:i+chunk_size] for i in range(0, len(model_base64), chunk_size)]

    # Create the embedded model weights string
    embedded_weights = 'EMBEDDED_MODEL_WEIGHTS = (\n'
    for chunk in chunks:
        embedded_weights += f'    "{chunk}"\n'
    embedded_weights += ')\n'

    print(f"Created {len(chunks)} chunks")

    # Find where to insert the embedded weights (after imports, before Config class)
    import_section_end = main_content.find('\n# ============================================================================\n# Configuration\n# ============================================================================')

    if import_section_end == -1:
        print("ERROR: Could not find Configuration section!")
        sys.exit(1)

    # Insert the embedded weights
    new_main_content = (
        main_content[:import_section_end] + 
        '\n\n# ============================================================================\n' +
        '# Embedded Model Weights\n' +
        '# ============================================================================\n\n' +
        embedded_weights +
        '\n' +
        main_content[import_section_end:]
    )

    # Modify the load_model method
    old_load_method = '''    def load_model(self, model_path: str):
        """Load trained model weights."""
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_loaded = True
            return True
        except Exception:
            return False'''

    new_load_method = '''    def load_model(self, model_path: str = None):
        """Load trained model weights from embedded data or file."""
        try:
            if model_path is None:
                # Load from embedded weights
                import base64
                import io
                model_bytes = base64.b64decode(EMBEDDED_MODEL_WEIGHTS)
                buffer = io.BytesIO(model_bytes)
                state_dict = torch.load(buffer, map_location=self.device, weights_only=True)
            else:
                # Load from file path
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_loaded = True
            return True
        except Exception:
            return False'''

    new_main_content = new_main_content.replace(old_load_method, new_load_method)

    # Update get_agent to try embedded weights first
    old_get_agent = '''def get_agent():
    """Get or create global agent instance."""
    global _agent
    if _agent is None:
        _agent = HybridAgent()
        # Try to load model from common paths
        paths = [
            '/kaggle_simulations/agent/best_model.pth',
            '/kaggle/input/connectx-v1/best_model.pth',
            'best_model.pth',
            './best_model.pth'
        ]
        for path in paths:
            if _agent.load_model(path):
                break
    return _agent'''

    new_get_agent = '''def get_agent():
    """Get or create global agent instance."""
    global _agent
    if _agent is None:
        _agent = HybridAgent()
        # Try to load from embedded weights first
        if not _agent.load_model():
            # Fallback to file paths
            paths = [
                '/kaggle_simulations/agent/best_model.pth',
                '/kaggle/input/connectx-v1/best_model.pth',
                'best_model.pth',
                './best_model.pth'
            ]
            for path in paths:
                if _agent.load_model(path):
                    break
    return _agent'''

    new_main_content = new_main_content.replace(old_get_agent, new_get_agent)

    # Save the new main.py
    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_main_content)

    print(f"\nCompleted!")
    print(f"  Original size: {len(main_content):,} bytes")
    print(f"  New size: {len(new_main_content):,} bytes")
    print(f"  Added: {len(new_main_content) - len(main_content):,} bytes")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create main.py with embedded model')
    parser.add_argument('--original', default='../submission/main_backup.py',
                        help='Path to original main.py')
    parser.add_argument('--model', default='model_weights_embedded.txt',
                        help='Path to base64 model file')
    parser.add_argument('--output', default='../submission/main.py',
                        help='Output path for new main.py')
    
    args = parser.parse_args()
    
    create_embedded_main(args.original, args.model, args.output)

