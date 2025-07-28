#!/bin/bash

ENV="env"
PYCACHE="**/__pychache"

if [ -d "$ENV" ]; then \
    rm -rf $ENV
    rm -rf $PYCACHE
    if env | grep -q "VIRTUAL_ENV="; then
        deactivate
    fi
    echo "Environnement virtuel supprimé."
else \
    echo "Auncun environnement virtuel détecté."
fi