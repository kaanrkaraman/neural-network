install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run black net tests
	poetry run isort net tests
	poetry run flake8 net tests