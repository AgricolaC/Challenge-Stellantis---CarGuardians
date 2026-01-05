import sys
import os
import subprocess
import importlib.metadata
import pkg_resources

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def check_dependencies(requirements_file='requirements.txt'):
    """Checks if dependencies in requirements.txt are installed."""
    print("Checking dependencies...")
    missing = []
    
    try:
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Basic parsing to get package name
                # Handles "numpy>=1.26.0,<2.0.0" -> "numpy"
                pkg_name = line.split('>')[0].split('<')[0].split('=')[0].strip()
                
                try:
                    importlib.metadata.version(pkg_name)
                except importlib.metadata.PackageNotFoundError:
                    missing.append(line)
    except FileNotFoundError:
        print(f"Error: {requirements_file} not found.")
        return []

    return missing

def install_dependencies(requirements_file='requirements.txt', force=False):
    """Installs dependencies via pip with improved build support."""
    print(f"Installing dependencies from {requirements_file}...")
    
    # 1. Upgrade build tools
    print("  Upgrading pip, setuptools, and wheel...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    
    # 2. Pre-install build dependencies (often fixes compilation issues on Mac/Py3.13)
    print("  Pre-installing numpy and cython for build isolation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.26.0", "cython"])

    # 3. Install main requirements
    cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_file]
    if force:
        cmd.append("--upgrade")
    
    subprocess.check_call(cmd)

def print_banner():
    print(r"""
   ______     C a r  G u a r d i a n s
  / ____/___ _____  _______  ____ _   __
 / /   / __ `/ __ \/ ___/ / / / _ \ | / /
/ /___/ /_/ / /_/ / /  / /_/ /  __/ |/ / 
\____/\__,_/_/ /_/_/   \__, /\___/|___/  
                      /____/             
    Scania APS Failure Prediction System
    """)

def main():
    clear_screen()
    print_banner()
    print("Welcome to the CarGuardians Environment Setup.")
    print("Please select your environment type:")
    print("1. Local Environment (Check cache first)")
    print("2. Virtual Environment (Install all dependencies)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    requirements_file = 'requirements.txt'
    
    if choice == '1':
        print("\n[Local Mode Selected]")
        missing = check_dependencies(requirements_file)
        if missing:
            print(f"\nFound {len(missing)} missing packages: {', '.join([m.split('[')[0].split('=')[0] for m in missing[:5]])}...")
            install = input("Install missing packages? (y/n): ").lower().strip()
            if install == 'y':
                install_dependencies(requirements_file)
        else:
            print("\nAll dependencies appear to be satisfied!")
            
    elif choice == '2':
        print("\n[Virtual Environment Mode Selected]")
        # In a real "Virtual Environment" setup, we might want to check if we are IN a venv.
        # check if sys.prefix != sys.base_prefix
        in_venv = (getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix) != sys.prefix
        if not in_venv:
             print("WARNING: You do not appear to be running inside a virtual environment.")
             proceed = input("Proceed anyway? (This will install to your global python) (y/n): ").lower().strip()
             if proceed != 'y':
                 print("Aborting.")
                 return
        
        install_dependencies(requirements_file, force=True)
    else:
        print("Invalid choice. Exiting.")
        return

    print("\n" + "="*50)
    print("       🚀 PIPELINE INSTRUCTIONS 🚀")
    print("="*50)
    
    print("\n1. 📊 Run Notebooks")
    print("   Command: jupyter notebook src/04_master_analysis.ipynb")
    
    print("\n2. 🕵️ Perform Forensics Analysis")
    print("   Command: python src/challenge/tests/test_ironclad.py")
    
    print("\n3. 🔍 Conduct Experimental RCA (Causal Discovery)")
    print("   Command: python src/challenge/experimental_rca/experimental_rca_pipeline.py")
    
    print("\n4. 📉 Execute Prediction Pipeline (Modelling)")
    print("   Command: python src/challenge/modelling/train_eval.py") # Assuming this is the script, or main driver
    
    print("\n")
    print("Setup Complete. Happy Coding!")

if __name__ == "__main__":
    main()
