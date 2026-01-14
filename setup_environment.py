"""
Setup Script for Thermal Camera Driver

This script helps set up the development environment for the thermal camera driver.

Usage:
    python setup_environment.py

Author: [Your Name]
Date: January 2026
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """
    Run a shell command and return success status.
    
    Args:
        command: Command to execute
        description: Description of the command
        
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("✗ Python 3.7 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True


def create_virtual_environment() -> bool:
    """Create a virtual environment."""
    venv_path = Path("thermal_camera_env")
    
    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True
    
    return run_command(
        f"{sys.executable} -m venv thermal_camera_env",
        "Creating virtual environment"
    )


def get_activation_command() -> str:
    """Get the appropriate activation command for the current OS."""
    if os.name == 'nt':  # Windows
        return r"thermal_camera_env\Scripts\activate"
    else:  # Unix/macOS
        return "source thermal_camera_env/bin/activate"


def install_requirements() -> bool:
    """Install required packages."""
    requirements_path = Path("requirements.txt")
    
    if not requirements_path.exists():
        print("✗ requirements.txt not found")
        return False
    
    # Get the appropriate pip command for the virtual environment
    if os.name == 'nt':  # Windows
        pip_command = r"thermal_camera_env\Scripts\pip"
    else:  # Unix/macOS
        pip_command = "thermal_camera_env/bin/pip"
    
    return run_command(
        f"{pip_command} install -r requirements.txt",
        "Installing required packages"
    )


def verify_installation() -> bool:
    """Verify that key packages are installed correctly."""
    if os.name == 'nt':  # Windows
        python_command = r"thermal_camera_env\Scripts\python"
    else:  # Unix/macOS
        python_command = "thermal_camera_env/bin/python"
    
    test_imports = [
        "import cv2",
        "import numpy",
        "import matplotlib.pyplot",
        "import zstandard",
        "import flirpy"
    ]
    
    print("\nVerifying package installation...")
    for import_cmd in test_imports:
        try:
            result = subprocess.run(
                [python_command, "-c", import_cmd],
                capture_output=True, text=True, check=True
            )
            package_name = import_cmd.split()[1].split('.')[0]
            print(f"✓ {package_name} imported successfully")
        except subprocess.CalledProcessError:
            package_name = import_cmd.split()[1].split('.')[0]
            print(f"✗ {package_name} import failed")
            return False
    
    return True


def create_directories() -> None:
    """Create necessary directories."""
    directories = ["recordings", "dual_data", "exported_frames"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def display_usage_instructions() -> None:
    """Display usage instructions."""
    activation_cmd = get_activation_command()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nTo use the thermal camera driver:")
    print("\n1. Activate the virtual environment:")
    if os.name == 'nt':
        print(f"   {activation_cmd}")
    else:
        print(f"   {activation_cmd}")
    
    print("\n2. Run the scripts:")
    print("   # Single camera recording:")
    print("   python record_thermal_video.py --output recording.npz --duration 30")
    print("\n   # Dual camera recording:")
    print("   python record_dual_thermal_video.py --output dual_recording.npz --duration 60")
    print("\n   # Live camera view:")
    print("   python wrapper_boson.py")
    print("\n   # Analyze recorded data:")
    print("   python analyze_data.py --input recording.npz --show-stats --show-video")
    
    print("\n3. Hardware requirements:")
    print("   - FLIR Boson thermal camera connected via USB")
    print("   - Proper camera drivers installed")
    print("   - For dual camera setup: cameras on COM4 and COM6")
    
    print(f"\n4. Deactivate virtual environment when done:")
    print("   deactivate")


def main() -> None:
    """Main setup function."""
    print("Thermal Camera Driver Environment Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("Failed to create virtual environment")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("Package verification failed")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Display usage instructions
    display_usage_instructions()


if __name__ == "__main__":
    main()