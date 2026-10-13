#!/bin/bash
# Sagi launcher — panggil Sagi dari mana saja dengan SUPER+S (lihat README).
# 100% model sendiri (Plosagata_More), tanpa model luar.
cd /home/rewsaqy/Projects/hoax-validator-12m || exit 1
exec .venv/bin/python scripts/pure_chat.py --size Plosagata_More
