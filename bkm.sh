#!/bin/sh
KMP_AFFINITY=scatter,granularity=fine th benchmark.lua
