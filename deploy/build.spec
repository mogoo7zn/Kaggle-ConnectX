# -*- mode: python ; coding: utf-8 -*-
import glob
import os

block_cipher = None

# Root path (one level up from the spec file location)
# PyInstaller injects SPECPATH, but we fallback to os.getcwd() if needed
try:
    spec_dir = SPECPATH
except NameError:
    spec_dir = os.getcwd()

root_path = os.path.abspath(os.path.join(spec_dir, '..'))

# Select specific files to include from submission folder
submission_files = []
# Add all python files from ../submission
for f in glob.glob(os.path.join(root_path, 'submission', '*.py')):
    # Store in 'submission' folder inside the app
    submission_files.append((f, 'submission'))

# Add specific model file
# Ensure we are picking up the correct file. 
# The user specified 'alpha-zero-ultra-weights.pth'
model_path = os.path.join(root_path, 'submission', 'alpha-zero-ultra-weights.pth')
if not os.path.exists(model_path):
    print(f"WARNING: Model file not found at {model_path}")
else:
    print(f"Including model file: {model_path}")
    submission_files.append((model_path, 'submission'))

a = Analysis(
    ['app.py'],
    pathex=[root_path],
    binaries=[],
    datas=[
        (os.path.join(root_path, 'agents'), 'agents'),
    ] + submission_files,
    hiddenimports=[
        'submission.main',
        'submission.main_alphazero',
        'submission.main_DQN',
        'submission.main_backup',
        'agents.base.config',
        'agents.base.utils',
        'torch',
        'numpy',
        'pygame'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ConnectX_Ultimate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ConnectX_Ultimate',
)
