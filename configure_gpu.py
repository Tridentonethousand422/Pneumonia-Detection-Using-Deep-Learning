import os
import sys

def main():
    # 1. Identify active virtual env path
    venv_path = sys.prefix
    if venv_path == sys.base_prefix:
        print("[ERROR] No virtual environment active. Please activate .venv-3.11 first!")
        sys.exit(1)

    # 2. Find site-packages
    site_packages = [p for p in sys.path if "site-packages" in p]
    if not site_packages:
        print("[ERROR] Could not locate site-packages directory.")
        sys.exit(1)
    
    site_packages_path = site_packages[0]
    nvidia_dir = os.path.join(site_packages_path, "nvidia")

    if not os.path.exists(nvidia_dir):
        print("[ERROR] NVIDIA package directory not found in site-packages.")
        print("        Please make sure you successfully ran: pip install \"tensorflow[and-cuda]\"")
        sys.exit(1)

    # 3. Find all lib sub-directories inside the nvidia package folder
    lib_paths = []
    for d in os.listdir(nvidia_dir):
        lib_dir = os.path.join(nvidia_dir, d, "lib")
        if os.path.isdir(lib_dir):
            lib_paths.append(lib_dir)

    # Also add the WSL driver pass-through directory
    lib_paths.append("/usr/lib/wsl/lib")

    # 4. Construct the LD_LIBRARY_PATH value
    ld_library_path_val = ":".join(lib_paths)

    # 5. Inject it into the activation script
    activate_script = os.path.join(venv_path, "bin", "activate")
    if not os.path.exists(activate_script):
        print(f"[ERROR] Could not find activation script at: {activate_script}")
        sys.exit(1)

    with open(activate_script, "r") as f:
        content = f.read()

    # Avoid duplicate injections
    if "LD_LIBRARY_PATH" in content:
        print("[INFO] Activation script already has LD_LIBRARY_PATH configuration. Skipping injection.")
    else:
        injection = (
            "\n# =========================================================================\n"
            "# AUTOMATIC TENSORFLOW GPU CONFIGURATION\n"
            "# =========================================================================\n"
            f"export LD_LIBRARY_PATH=\"{ld_library_path_val}:$LD_LIBRARY_PATH\"\n"
            "# =========================================================================\n"
        )
        with open(activate_script, "a") as f:
            f.write(injection)
        print(f"[SUCCESS] Injected GPU libraries path successfully into: {activate_script}")

    print("\n[IMPORTANT] To apply these changes:")
    print("            1. Deactivate current session:  deactivate")
    print("            2. Reactivate virtual env:      source .venv-3.11/bin/activate")

if __name__ == "__main__":
    main()
