import importlib

print("Testing import...")

try:
    mod = importlib.import_module("src.training.trainer")
    print("Trainer imported successfully:", mod)
except Exception as e:
    print("Import error:")
    print(e)
