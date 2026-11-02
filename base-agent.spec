# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building base-agent into a macOS .app bundle.

Usage:
    pyinstaller base-agent.spec

Produces:
    dist/BaseAgent.app       — macOS application bundle
    dist/BaseAgent           — command-line executable
"""

import os
import sys
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

block_cipher = None

# -----------------------------------------------------------------------
# Collect all data files and submodules for complex dependencies
# -----------------------------------------------------------------------

datas = []
binaries = []
hiddenimports = []

# --- PyQt6 ---
datas += collect_data_files('PyQt6')
hiddenimports += collect_submodules('PyQt6')

# --- httpx / httpcore ---
hiddenimports += collect_submodules('httpx')
hiddenimports += collect_submodules('httpcore')
hiddenimports += collect_submodules('h11')
hiddenimports += collect_submodules('certifi')

# --- BeautifulSoup4 ---
datas += collect_data_files('bs4')

# --- LanceDB (vector store) ---
hiddenimports += collect_submodules('lancedb')
datas += collect_data_files('lancedb')
# LanceDB uses pyarrow / lance under the hood
hiddenimports += collect_submodules('pyarrow')
datas += collect_data_files('pyarrow')

# --- FastEmbed (embedding model runner) ---
hiddenimports += collect_submodules('fastembed')
datas += collect_data_files('fastembed')
# onnxruntime is used by fastembed
hiddenimports += collect_submodules('onnxruntime')
datas += collect_data_files('onnxruntime')

# --- Reranker 现由 fastembed 提供，无需 FlagEmbedding hiddenimport ---

# --- Document parsers (optional RAG deps) ---
for mod in ['docx', 'openpyxl', 'pptx', 'fitz', 'ebooklib']:
    try:
        hiddenimports += collect_submodules(mod)
        datas += collect_data_files(mod)
    except Exception:
        pass

# --- docling (unified document parser, lazy-loaded) ---
try:
    hiddenimports += collect_submodules('docling')
    datas += collect_data_files('docling')
except Exception:
    pass

# --- chonkie (semantic chunking, lazy-loaded) ---
try:
    hiddenimports += collect_submodules('chonkie')
    datas += collect_data_files('chonkie')
except Exception:
    pass

# --- jieba (BM25 CJK tokenizer) ---
try:
    hiddenimports += collect_submodules('jieba')
    datas += collect_data_files('jieba')
except Exception:
    pass

# --- numpy (semantic chunker vector ops) ---
try:
    hiddenimports += collect_submodules('numpy')
except Exception:
    pass

# --- RapidOCR (optional OCR for scanned PDFs) ---
try:
    hiddenimports += collect_submodules('rapidocr_onnxruntime')
    datas += collect_data_files('rapidocr_onnxruntime')
except Exception:
    pass

# --- Package metadata (needed by some libs for version checks) ---
datas += copy_metadata('PyQt6')
datas += copy_metadata('httpx')
datas += copy_metadata('lancedb')
datas += copy_metadata('fastembed')
datas += copy_metadata('beautifulsoup4')
# docling / chonkie 为可选依赖（未安装时回退到按格式解析/递归分片），
# 用 try/except 保护，避免打包时因缺少元数据而失败。
for _pkg in ('docling', 'chonkie'):
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        pass

# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------

a = Analysis(
    ['main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused modules to reduce bundle size
        'matplotlib',
        'numpy.tests',
        'pandas.tests',
        'pytest',
        'IPython',
        'notebook',
        'jupyter',
        'tkinter',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# -----------------------------------------------------------------------
# Command-line executable (single file)
# -----------------------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BaseAgent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Keep console for log output
)

# -----------------------------------------------------------------------
# macOS .app bundle
# -----------------------------------------------------------------------

app = BUNDLE(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='BaseAgent.app',
    icon='BaseAgent.icns',
    bundle_identifier='com.baseagent.app',
    info_plist={
        'CFBundleDisplayName': 'BaseAgent',
        'CFBundleShortVersionString': '1.2.8',
        'CFBundleVersion': '1',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'BaseAgent may need microphone access for voice input.',
        'LSMinimumSystemVersion': '12.0',
    },
)
