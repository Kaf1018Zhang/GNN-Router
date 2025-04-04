#!/bin/bash
# pip install -r requirements.txt

echo "==== Train controller ===="
python experiments/train_controller.py

echo "==== Deploy controller ===="
python experiments/deploy_controller.py
