# Packaging ConnectX Ultimate

This folder contains the scripts and configuration to package the game into a standalone application.

## Prerequisites

1.  Python 3.8+ installed.
2.  Install dependencies (from root directory):
    ```bash
    pip install -r ../requirements.txt
    pip install pyinstaller
    ```

## Building the App

### Windows

Run the `package.bat` script in this directory:

```bash
.\package.bat
```

The executable will be in `dist\ConnectX_Ultimate\ConnectX_Ultimate.exe`.

### macOS / Linux

Open a terminal in this directory (`deploy/`) and run:

```bash
pyinstaller build.spec
```

The executable will be in `dist/ConnectX_Ultimate/ConnectX_Ultimate`.

## Troubleshooting

- **Missing Imports**: If the app crashes immediately, check the console output (change `console=False` to `console=True` in `build.spec` to debug).
- **Large Size**: The app includes PyTorch, which is large. This is normal.
