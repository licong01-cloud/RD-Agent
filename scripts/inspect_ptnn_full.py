#!/usr/bin/env python3
"""Inspect GeneralPTNN full training/validation loop."""
import qlib.contrib.model.pytorch_general_nn as ptnn
import inspect

src = inspect.getsource(ptnn.GeneralPTNN)
print(src)
