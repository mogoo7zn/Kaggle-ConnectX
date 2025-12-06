# 📜 Scripts Directory

This directory contains automation scripts for setting up the ConnectX project environment.

## 📋 Available Scripts

### `setup_env.bat` (Windows)

Automated environment setup script for Windows systems.

**Features:**
- ✅ Checks Python version (requires 3.8+)
- ✅ Creates Python virtual environment (`venv/`)
- ✅ Installs all dependencies from `requirements.txt`
- ✅ Provides clear feedback and error handling

**Usage:**
```bash
scripts\setup_env.bat
```

**What it does:**
1. Verifies Python installation
2. Creates or recreates virtual environment
3. Activates the virtual environment
4. Upgrades pip to latest version
5. Installs all project dependencies

### `setup_env.sh` (Linux/Mac)

Automated environment setup script for Linux and macOS systems.

**Features:**
- ✅ Checks Python version (requires 3.8+)
- ✅ Creates Python virtual environment (`venv/`)
- ✅ Installs all dependencies from `requirements.txt`
- ✅ Handles existing virtual environments gracefully
- ✅ Provides clear feedback and error handling

**Usage:**
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

**What it does:**
1. Verifies Python 3 installation
2. Prompts before recreating existing virtual environment
3. Creates virtual environment
4. Activates the virtual environment
5. Upgrades pip to latest version
6. Installs all project dependencies

## 🚀 Quick Start

After running the setup script, activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

Then you can start using the project:
```bash
# Play the game
python playground/play.py

# Train agents
python -m agents.rainbow.train_rainbow
python -m agents.alphazero.train_alphazero
```

## 📦 Dependencies Installed

The scripts install all dependencies from `requirements.txt`, including:

- **PyTorch** (>=2.0.0) - Deep learning framework
- **NumPy** (>=1.24.0) - Numerical computing
- **Matplotlib** (>=3.7.0) - Visualization
- **Pygame** (>=2.5.0) - Game interface for playground
- **TensorBoard** (>=2.13.0) - Training monitoring
- **tqdm** (>=4.65.0) - Progress bars

## 🔧 Troubleshooting

### Script fails to create virtual environment

**Solution:** Ensure Python 3.8+ is installed and accessible in PATH:
```bash
python --version  # Should show 3.8 or higher
```

### Permission denied (Linux/Mac)

**Solution:** Add execute permission to the script:
```bash
chmod +x scripts/setup_env.sh
```

### Virtual environment already exists

**Windows:** The script will prompt you to recreate it. Type `y` to recreate or `N` to skip.

**Linux/Mac:** The script will prompt you. Type `y` to recreate or press Enter to skip.

### Dependencies fail to install

**Solution:** 
1. Ensure you have internet connection
2. Try upgrading pip first: `python -m pip install --upgrade pip`
3. Check if there are any specific error messages
4. For PyTorch with CUDA, you may need to install it separately:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## 📝 Notes

- The virtual environment is created in the project root as `venv/`
- The `venv/` directory should be added to `.gitignore` (not committed to version control)
- After setup, always activate the virtual environment before running project scripts
- To deactivate: simply run `deactivate` in the terminal

## 🔄 Recreating the Environment

If you need to recreate the virtual environment:

**Windows:**
```bash
rmdir /s /q venv
scripts\setup_env.bat
```

**Linux/Mac:**
```bash
rm -rf venv
./scripts/setup_env.sh
```

---

**Happy coding! 🎉**

