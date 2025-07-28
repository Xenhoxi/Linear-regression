#!/bin/bash

ENV="env"

if [ ! -d "$ENV" ]; then \
    python3 -m venv $ENV && \
    source $ENV/bin/activate
    $ENV/bin/pip install -r requirements.txt; \
    echo "Environnement virtuel créé avec les dépendances."; \
else \
    if env | grep -q "VIRTUAL_ENV="; then
        echo "L'environnement virtuel est déjà actif."; \
    else
        source $ENV/bin/activate
        echo "L'environnement virtuel existé déjà. Mais viens d'etre activé."; \
    fi
    
fi