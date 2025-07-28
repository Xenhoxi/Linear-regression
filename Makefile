# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ljerinec <ljerinec@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/04 16:29:17 by ljerinec          #+#    #+#              #
#    Updated: 2025/02/04 16:49:41 by ljerinec         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

.PHONY: venv clean requirements

VENV = env

venv:
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV) && \
		$(VENV)/bin/pip install -r requirements.txt; \
		echo "Environnement virtuel créé avec les dépendances."; \
	else \
		echo "L'environnement virtuel existe déjà."; \
	fi

clean:
	@rm -rf $(VENV)
	@echo "Environnement virtuel supprimé."
	@rm -rf training_result.csv
	@rm -rf Plot.png

requirements:
	@if [ -d "$(VENV)" ]; then \
		$(VENV)/bin/pip freeze > requirements.txt; \
		echo "requirements.txt mis à jour."; \
	else \
		echo "Erreur: L'environnement virtuel n'existe pas. Exécutez 'make venv' d'abord." >&2; \
		exit 1; \
	fi