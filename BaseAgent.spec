# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BaseAgent macOS .app bundle."""

import sys
from pathlib import Path

a = Analysis(
    ['main.py'],
    pathex=[str(Path.cwd())],
    binaries=[],
    datas=[
        ('src', 'src'),  # bundle the entire source tree
    ],
    hiddenimports=[
        # PyQt6
        'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
        # RAG stack (lazy-loaded, PyInstaller won't detect them automatically)
        'lancedb', 'fastembed', 'fitz',  # PyMuPDF
        'docx', 'openpyxl', 'pptx',
        'bs4', 'ebooklib',
        # fastembed internals
        'fastembed.text.onnx_embedding',
        'fastembed.text.text_embedding',
        'fastembed.common.model_management',
        # huggingface_hub (used by fastembed)
        'huggingface_hub',
        # rapidocr (optional OCR)
        'rapidocr_onnxruntime',
        # Reranker 现由 fastembed 提供，无需单独 hiddenimport
        # MCP
        'mcp',
        # httpx
        'httpx',
        # agent internal modules
        'agent', 'agent.ui', 'agent.ui.main_window',
        'agent.rag', 'agent.rag.engine', 'agent.rag.vector_store',
        'agent.rag.parsers', 'agent.rag.deps',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'unittest', 'test', 'tests',
        'matplotlib', 'pandas', 'scipy',
        'notebook', 'jupyter', 'ipykernel',
        'pytest', 'coverage',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BaseAgent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # macOS GUI app, no terminal window
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
    name='BaseAgent',
)

app = BUNDLE(
    coll,
    name='BaseAgent.app',
    icon='icon.icns',
    bundle_identifier='com.baseagent.app',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15.0',
        'CFBundleName': 'BaseAgent',
        'CFBundleDisplayName': 'BaseAgent',
        'CFBundleShortVersionString': '1.2.18',
        'CFBundleVersion': '1.2.18',
        'CFBundleExecutable': 'BaseAgent',
        'CFBundlePackageType': 'APPL',
        'NSHumanReadableCopyright': 'BaseAgent - Local AI Agent',
    },
)
