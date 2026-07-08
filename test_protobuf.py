try:
    from google.protobuf import runtime_version
    print("Successfully imported runtime_version")
except ImportError as e:
    print(f"ImportError (google.protobuf): {e}")

try:
    from transformers import pipeline
    print("Successfully imported transformers pipeline")
except ImportError as e:
    print(f"ImportError (transformers): {e}")
except Exception as e:
    print(f"Exception (transformers): {e}")
