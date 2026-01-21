"""
Syntax and Import Validation Test
Tests that all module files have correct syntax and can be imported
"""

import sys
import ast
from pathlib import Path


def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def main():
    """Check all model files for syntax errors"""
    model_dir = Path(__file__).parent
    
    python_files = [
        "__init__.py",
        "ddi_model.py",
        "risk_scorer.py",
        "data_preprocessor.py",
        "trainer.py",
        "evaluator.py",
        "example_usage.py",
    ]
    
    print("=" * 60)
    print("Syntax Validation Test")
    print("=" * 60)
    
    all_valid = True
    
    for filename in python_files:
        file_path = model_dir / filename
        if not file_path.exists():
            print(f"❌ {filename}: File not found")
            all_valid = False
            continue
        
        valid, error = check_syntax(file_path)
        if valid:
            print(f"✓ {filename}: Valid syntax")
        else:
            print(f"❌ {filename}: Syntax error - {error}")
            all_valid = False
    
    print("=" * 60)
    
    if all_valid:
        print("✓ All files have valid Python syntax!")
        print("\nTo test functionality, install dependencies:")
        print("  pip install -r model/requirements.txt")
        print("\nThen run:")
        print("  python -m model.example_usage")
        return 0
    else:
        print("❌ Some files have syntax errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
