import sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path

from materials_data_lab.model_artifact import build_or_load_artifact
from materials_data_lab.demo import create_demo

def main():
    parser = argparse.ArgumentParser(description="Launch Materials Data Lab Local Demo")
    parser.add_argument("--csv", type=str, default="data/raw/Tempering data for carbon and low alloy steels - Raiipa.csv", help="Path to raw CSV file")
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    print("Loading or building model artifact...")
    models, meta = build_or_load_artifact(csv_path)
    
    if meta.get("retrained"):
        print("Models were retrained from data.")
    else:
        print("Loaded existing model artifact.")
        
    demo = create_demo(models, meta)
    
    # Launch strictly locally without share=True
    demo.launch(inbrowser=True, share=False)

if __name__ == "__main__":
    main()
