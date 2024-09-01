#!/bin/bash

set -e

cd "$(dirname "$0")/../apps"

python -m pytest ../tests/
