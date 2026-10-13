#!/bin/bash
# Sagi launcher — panggil Sagi dari mana saja dengan SUPER+S (lihat README).
# Mode dummy: 100% kode sendiri (template + model 12M), tanpa download model luar.
cd /home/rewsaqy/Projects/hoax-validator-12m || exit 1
exec .venv/bin/python chat.py --dummy
