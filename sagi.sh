#!/bin/bash
# Sagi launcher — panggil Sagi dari mana saja dengan SUPER+S (lihat README).
# 100% model sendiri (Plosagata b1), tanpa model luar.
cd /home/rewsaqy/Projects/hoax-validator-12m || exit 1
exec .venv/bin/python pure_chat.py --size b1
